import time
import unittest
from threading import Thread
from typing import Any, Dict, Union

import torch
from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
from torch import nn

from buglab.data.modelsync import ModelSyncClient, ModelSyncServer


class MockNeuralModule(ModuleWithMetrics):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        return self.param + data


class MockNeuralModel(AbstractNeuralModel[int, int, MockNeuralModule]):
    def __init__(self, some_hyper: int):
        super().__init__()
        self.some_hyper = some_hyper

    def update_metadata_from(self, datapoint: int) -> None:
        pass

    def build_neural_module(self) -> MockNeuralModule:
        return MockNeuralModule()

    def tensorize(self, datapoint: int) -> int:
        return datapoint

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {"data": []}

    def extend_minibatch_with(self, tensorized_datapoint: int, partial_minibatch: Dict[str, Any]) -> bool:
        partial_minibatch["data"].append(tensorized_datapoint)

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        return {"data": torch.tensor(accumulated_minibatch_data["data"], device=device)}


class TestModelSync(unittest.TestCase):
    def __create_mock_model_and_nn(self, some_int):
        model = MockNeuralModel(some_int)
        neural_net = model.build_neural_module()
        return model, neural_net

    def test_model_sync1(self):
        """Note that this test should be run in a separate process from the next one."""
        model, neural_net = self.__create_mock_model_and_nn(0)
        server = ModelSyncServer("tcp://*:9000", model, neural_net)

        client = ModelSyncClient("tcp://localhost:9000")
        model_remote, neural_net_remote = client.ask("cpu")
        self.assertEqual(model_remote.some_hyper, 0)
        self.assertEqual(int(neural_net_remote.param), 0)

        # Emulate parameter update
        with torch.no_grad():
            neural_net.param.add_(1)
        self.assertEqual(int(neural_net.param), 1)
        server.update_parameters(neural_net)

        client.update_params_if_needed(neural_net_remote, "cpu")
        self.assertEqual(int(neural_net_remote.param), 1)

        # Server has no updates
        client.update_params_if_needed(neural_net_remote, "cpu")
        self.assertEqual(int(neural_net_remote.param), 1)

    def __emulate_slow_model_startup(self):
        time.sleep(10)
        model, neural_net = self.__create_mock_model_and_nn(0)
        server = ModelSyncServer("tcp://*:9000", model, neural_net)

    def test_model_sync2(self):
        t = Thread(target=self.__emulate_slow_model_startup)
        t.start()

        # The server will be ready in approximately 10'
        client = ModelSyncClient("tcp://localhost:9000")
        model_remote, neural_net_remote = client.ask("cpu")

        # Did we ever get here?
        self.assertEqual(model_remote.some_hyper, 0)
        self.assertEqual(int(neural_net_remote.param), 0)


if __name__ == "__main__":
    unittest.main()
