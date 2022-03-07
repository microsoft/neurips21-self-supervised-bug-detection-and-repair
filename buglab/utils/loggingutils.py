import logging
import numpy as np
import os
from contextlib import ExitStack, contextmanager
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerSpanExporter as SpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchExportSpanProcessor
from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    push_to_gateway,
    start_http_server,
)
from queue import Queue


def configure_logging():
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s [%(name)-35.35s @ %(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    )


class LatencyRecorder:
    def __init__(self, latency_recorder: Histogram):
        self._latency_recorder = latency_recorder
        self._timer = latency_recorder.time()
        self._exit_stack = ExitStack()
        self._running = False

    def start(self):
        assert not self._running, "Tried to start a latency recorder that is already running."
        self._running = True
        self._exit_stack.enter_context(self._timer)

    def stop(self):
        assert self._running, "Tried to stop a latency recorder that is not running."
        self._running = False
        self._exit_stack.close()

    def __enter__(self):
        self._timer.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._timer.__exit__(exc_type, exc_val, exc_tb)


class ValueRecorder:
    def __init__(self, value_recorder: Summary):
        self._value_recorder = value_recorder

    def record(self, value: float):
        self._value_recorder.observe(value)


class MetredQueue(Queue):
    def __init__(self, gauge: Gauge, maxsize=0):
        self._gauge = gauge
        super().__init__(maxsize=maxsize)

    def put(self, item, block=True, timeout=None):
        super().put(item, block=block, timeout=timeout)
        self._gauge.inc()

    def get(self, block=True, timeout=None):
        item = super().get(block=block, timeout=timeout)
        self._gauge.dec()
        return item


class MetricProvider:

    _histogram_boundaries = list(np.linspace(start=0.001, stop=128, num=512)) + [256, 512]
    _tracing_set_up = False

    def __init__(self, module_name: str, push: bool = False, push_gateway_address: str = "localhost:9091"):
        self._module_name = module_name.replace(".", "_") + "_"  # Grafana cannot handle '.' in metric names.
        self._server_started = False
        self._tracing = False
        self._push = push
        self._registry = REGISTRY
        if self._push:
            self._registry = CollectorRegistry()
        self._push_gateway = push_gateway_address

    def start_server(self, port: int = 8000) -> None:
        assert not self._push, "Should not start server if we are using the push gateway."
        try:
            start_http_server(port)
            self._server_started = True
        except OSError as e:
            print("Error trying to start metric logging. Try a different port.")
            raise e

    def set_push_gateway_address(self, address: str) -> None:
        assert self._push, "Trying to set a push gateway address when not using push metrics."
        self._push_gateway = address

    def set_tracing(self, tracing_on: bool) -> None:
        self._tracing = tracing_on

    def _check_ready(self):
        assert (
            self._server_started or self._push
        ), "The Prometheus server needs to start before metrics are created, or push must be enabled."

    def new_counter(self, counter_name: str, description: str = "") -> Counter:
        self._check_ready()
        counter = Counter(name=self._module_name + counter_name, documentation=description, registry=self._registry)
        return counter

    def new_gauge(self, gauge_name: str, description: str = "") -> Gauge:
        self._check_ready()
        return Gauge(name=self._module_name + gauge_name, documentation=description, registry=self._registry)

    def new_queue(self, queue_name: str, maxsize=0, description: str = "") -> MetredQueue:
        self._check_ready()
        gauge = self.new_gauge(gauge_name=queue_name, description=description)
        return MetredQueue(gauge=gauge, maxsize=maxsize)

    def new_latency_measure(self, measure_name: str, description: str = "") -> LatencyRecorder:
        self._check_ready()
        latency_measure = Histogram(
            name=self._module_name + measure_name,
            documentation=description,
            buckets=self._histogram_boundaries,
            registry=self._registry,
        )
        return LatencyRecorder(latency_measure)

    def new_measure(self, measure_name: str, description: str = "") -> ValueRecorder:
        self._check_ready()
        return ValueRecorder(
            Summary(name=self._module_name + measure_name, documentation=description, registry=self._registry)
        )

    def get_tracer(self) -> trace.Tracer:
        if not self._tracing:
            return DummyTracer()

        if not self._tracing_set_up:
            jaeger_exporter = SpanExporter(
                service_name=self._module_name,
                agent_host_name="localhost",
                agent_port=6831,
            )
            trace.set_tracer_provider(TracerProvider())
            trace.get_tracer_provider().add_span_processor(BatchExportSpanProcessor(jaeger_exporter))
            self._tracing_set_up = True
        return trace.get_tracer(self._module_name)

    def push_metrics(self, job: str):
        assert self._push, "Metric provider not configured for push."
        push_to_gateway(self._push_gateway, job, self._registry)


class DummyTracer:
    @contextmanager
    def start_as_current_span(*args, **kwargs):
        yield
