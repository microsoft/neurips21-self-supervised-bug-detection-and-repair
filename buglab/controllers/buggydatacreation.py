from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import jedi
import libcst as cst
import logging
import msgpack
import multiprocessing
import os
import zmq
from copy import deepcopy
from jedi.api.environment import Environment
from libcst.metadata import CodeRange
from os import PathLike
from pathlib import Path

from buglab.data.deduplication import DuplicationClient
from buglab.data.deduplication.tokenizers import python_dedup_tokenize_file
from buglab.data.pypi.venv import create_venv_and_install
from buglab.representations.coderelations import compute_all_relations
from buglab.representations.codereprs import DummyEntity, PythonCodeRelations
from buglab.representations.hypergraph import convert_buglab_sample_to_hypergraph
from buglab.rewriting import (
    ALL_REWRITE_SCOUTS,
    AbstractRewriteOp,
    ICodeRewriteScout,
    apply_semantics_preserving_transforms,
    filter_ops_in_range,
)
from buglab.utils import call_with_timeout, detect_encoding_and_open
from buglab.utils.cstutils import AllFunctionFinder, subsumes_code_range
from buglab.utils.loggingutils import LatencyRecorder, MetricProvider

LOGGER = logging.getLogger(__name__)
metric_provider = MetricProvider("BuggyDataCreator")


def get_serialized_representation(
    rel_db: PythonCodeRelations,
    target_range: CodeRange,
    available_rewrite_ops: List[AbstractRewriteOp],
    available_rewrite_op_metadata: List[Tuple[Type[ICodeRewriteScout], Any]],
    target_fix_op: Optional[AbstractRewriteOp],
    package_name: str,
    package_version: str,
):
    if target_fix_op is not None:
        target_action_idx = available_rewrite_ops.index(target_fix_op)
        if target_action_idx == -1:
            LOGGER.error("Found non reversible op for %s at %s (%s)", package_name, rel_db.path, target_range)
            return None
    else:
        target_action_idx = None

    serializable, node_to_idx = rel_db.as_serializable(
        target_range=target_range, reference_nodes=[ref_node for _, ref_node, _ in available_rewrite_op_metadata]
    )

    def metadata_as_serializable(metadata: Any):
        if isinstance(metadata, (cst.CSTNode, DummyEntity)):
            return node_to_idx[metadata]
        elif isinstance(metadata, str):
            return node_to_idx[metadata]
        return metadata

    return {
        "graph": serializable,
        "candidate_rewrites": [(op.op_name(), op.rewrite_data()) for op in available_rewrite_ops],
        "candidate_rewrite_ranges": [
            (
                (op.target_range.start.line, op.target_range.start.column),
                (op.target_range.end.line, op.target_range.end.column),
            )
            for op in available_rewrite_ops
        ],
        "candidate_rewrite_metadata": [
            (cls.__name__, metadata_as_serializable(n)) for cls, _, n in available_rewrite_op_metadata
        ],
        "target_fix_action_idx": target_action_idx,
        "package_name": package_name,
        "package_version": package_version,
    }


def apply_rewrite_and_create_sample(
    code_text: str,
    rewrite_op: AbstractRewriteOp,
    filepath: PathLike,
    package_name: str,
    package_version: str,
    jedi_env: Environment,
):
    modified_code_text, reverse_op = rewrite_op.rewrite(code_text)

    rel_db = PythonCodeRelations(modified_code_text, Path(filepath))
    compute_all_relations(rel_db, jedi_env)

    function_finder = AllFunctionFinder()
    rel_db.ast_with_metadata_wrapper.visit(function_finder)

    for function, fn_pos in function_finder.all_function_nodes:
        if subsumes_code_range(reverse_op.target_range, fn_pos):
            target_fn_pos = fn_pos
            break
    else:
        LOGGER.error("Could not find target function for %s %s %s", reverse_op, filepath, package_name)
        return None

    available_ops: List[AbstractRewriteOp] = []
    available_ops_metadata: List[Tuple[Type[ICodeRewriteScout], cst.CSTNode, Any]] = []
    rel_db.ast_with_metadata_wrapper.visit_batched(
        [ScoutClass(available_ops, available_ops_metadata) for ScoutClass in ALL_REWRITE_SCOUTS]
    )
    relevant_ops, relevant_op_metadata = filter_ops_in_range(available_ops, available_ops_metadata, target_fn_pos)

    if reverse_op not in relevant_ops:
        LOGGER.error("Found non-reversible op for %s at %s (%s)", package_name, rel_db.path, rewrite_op)
        return None

    return get_serialized_representation(
        rel_db,
        target_fn_pos,
        relevant_ops,
        relevant_op_metadata,
        target_fix_op=reverse_op,
        package_name=package_name,
        package_version=package_version,
    )


def extract_bugs_from(
    filepath: PathLike,
    venv_location: str,
    package_name: str,
    package_version: str,
    rewrite_selector_server_address: str,
    request_latency_measure: LatencyRecorder,
    compute_relations_timeout_sec: float = 60,
    num_semantics_preserving_transformations: int = 1,
) -> Iterator[Dict]:
    # Establish connection with Bug Selector Server
    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.connect(rewrite_selector_server_address)

    jedi.settings.cache_directory = os.path.join(
        venv_location, f".cache_{multiprocessing.current_process().pid}", "jedi"
    )
    jedi.settings.call_signatures_validity = 60 * 2
    jedi_env = jedi.create_environment(venv_location)

    LOGGER.info("Introducing and extracting random bugs from %s", filepath)
    with detect_encoding_and_open(filepath) as f:
        original_code_text = f.read()

    input_snippets = {original_code_text}

    # If configured, also generate different variants of the input:
    if num_semantics_preserving_transformations > 0:
        num_attempts = 0
        while len(input_snippets) - 1 <= num_semantics_preserving_transformations:
            if num_attempts > 2 * num_semantics_preserving_transformations:
                # We are not making progress - maybe snippet doesn't admit transforms, or
                # p_apply_each in apply_semantics_preserving_transforms is configured too low.
                LOGGER.warning(
                    f"Failed to generate {num_semantics_preserving_transformations}"
                    f" semantically equivalent variants of snippet after {num_attempts}."
                )
                break
            num_attempts += 1
            try:
                input_snippets.add(apply_semantics_preserving_transforms(original_code_text))
            except Exception as e:
                LOGGER.exception("Error in applying semantics-preserving transforms", exc_info=e)

    for code_text in input_snippets:
        rel_db = PythonCodeRelations(code_text, Path(filepath))
        try:
            call_with_timeout(lambda: compute_all_relations(rel_db, jedi_env), timout_sec=compute_relations_timeout_sec)
        except TimeoutError:
            LOGGER.error("Computing Entity Relations in %s of `%s` timed out. Aborting.", filepath, package_name)
            return

        function_finder = AllFunctionFinder()
        rel_db.ast_with_metadata_wrapper.visit(function_finder)

        available_ops: List[AbstractRewriteOp] = []
        available_ops_metadata: List[Tuple[Type[ICodeRewriteScout], cst.CSTNode, Any]] = []
        rel_db.ast_with_metadata_wrapper.visit_batched(
            [ScoutClass(available_ops, available_ops_metadata) for ScoutClass in ALL_REWRITE_SCOUTS]
        )

        for function, fn_pos in function_finder.all_function_nodes:
            relevant_ops, relevant_op_metadata = filter_ops_in_range(available_ops, available_ops_metadata, fn_pos)
            if len(relevant_ops) > 5:  # To avoid introducing bias
                serialized_original_code = get_serialized_representation(
                    rel_db, fn_pos, relevant_ops, relevant_op_metadata, None, package_name, package_version
                )
                # RPC to rewrite selector server
                with request_latency_measure:
                    socket.send(msgpack.dumps(serialized_original_code))
                    selected_rewrites: Dict[Optional[str], float] = msgpack.loads(socket.recv())

                function_graph_and_rewrites = {"original": serialized_original_code, "rewrites": {}}

                for rewrite_idx, rewrite_prob in selected_rewrites.items():
                    if rewrite_idx == "NO_BUG":
                        # No rewrite
                        function_graph_and_rewrites["rewrites"]["NO_BUG"] = (serialized_original_code, rewrite_prob)
                        continue

                    rewrite_op = relevant_ops[int(rewrite_idx)]
                    try:
                        rewritten_graph = apply_rewrite_and_create_sample(
                            code_text, rewrite_op, filepath, package_name, package_version, jedi_env
                        )

                        function_graph_and_rewrites["rewrites"][rewrite_idx] = (rewritten_graph, rewrite_prob)
                    except Exception as e:
                        LOGGER.exception(
                            "Failed when applying %s at %s (package: %s)",
                            rewrite_op,
                            filepath,
                            package_name,
                            exc_info=e,
                        )
                yield function_graph_and_rewrites


def should_extract(
    package_name: str, file: str, filepath: str, deduplication_client: Optional[DuplicationClient] = None
) -> bool:
    if not filepath.endswith(".py"):
        return False
    if deduplication_client is not None:
        if deduplication_client.check_if_duplicate_and_add(
            f"{package_name}::{file}", list(set(python_dedup_tokenize_file(filepath)["tokens"]))
        ):
            LOGGER.info("Excluding %s::%s as a duplicate.", package_name, filepath)
            return False
    return True


def extract_for_package(
    package_name: str,
    bug_selector_server_address: str,
    push_gateway_address: Optional[str] = None,
    deduplication_client: Optional[DuplicationClient] = None,
    num_semantics_preserving_transformations_per_file: int = 1,
    as_hypergraph: bool = False,
) -> Iterator:
    """
    :param package_name: The name of the package.
    :param bug_selector_server_address: The address of the rewrite selector server.
    :param push_gateway_address: The address of the Prometheus push gateway.
    :param deduplication_client: A link to the deduplication server.
    :param num_semantics_preserving_transformations_per_file: the number of semantics-preserving transforms to
        make to each input file.
    """
    # Get logging set up for this function.
    if push_gateway_address is not None:
        metric_provider.set_push_gateway_address(push_gateway_address)
    else:
        metric_provider.start_server(8001)  # TODO: Less hard-coded...

    package_timer = metric_provider.new_latency_measure("package_timer")
    request_latency_measure = metric_provider.new_latency_measure("request_latency")
    samples_published = metric_provider.new_counter("samples_published")

    with create_venv_and_install(package_name) as venv, package_timer:
        for file in venv.all_package_files:
            filepath = os.path.join(venv.package_location, file)
            if not should_extract(package_name, file, filepath, deduplication_client):
                continue

            extracted_data = extract_bugs_from(
                filepath=filepath,
                venv_location=venv.venv_location,
                package_name=package_name,
                package_version=venv.package_version,
                rewrite_selector_server_address=bug_selector_server_address,
                request_latency_measure=request_latency_measure,
                num_semantics_preserving_transformations=num_semantics_preserving_transformations_per_file,
            )
            for ex in extracted_data:
                samples_published.inc()
                if as_hypergraph:
                    ex = {
                        "original": convert_buglab_sample_to_hypergraph(deepcopy(ex["original"])),
                        "rewrites": {
                            k: (convert_buglab_sample_to_hypergraph(deepcopy(g)), p)
                            for k, (g, p) in ex["rewrites"].items()
                        },
                    }
                yield ex
