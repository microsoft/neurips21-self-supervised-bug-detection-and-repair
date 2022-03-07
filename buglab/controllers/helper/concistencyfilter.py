from typing import Any, List, NamedTuple, Optional, Tuple, Type

import argparse
import jedi
import libcst as cst
import math
import numpy as np
import torch
import traceback
from dpu_utils.utils import run_and_debug
from enum import Enum, auto
from pathlib import Path
from tqdm import tqdm

from buglab.controllers.buggydatacreation import get_serialized_representation
from buglab.models.gnn import GnnBugLabModel
from buglab.representations.coderelations import compute_all_relations
from buglab.representations.codereprs import PythonCodeRelations
from buglab.representations.hypergraph import convert_buglab_sample_to_hypergraph
from buglab.rewriting import ALL_REWRITE_SCOUTS, AbstractRewriteOp, ICodeRewriteScout, filter_ops_in_range
from buglab.utils import detect_encoding_and_open
from buglab.utils.cstutils import AllFunctionFinder
from buglab.utils.loggingutils import configure_logging


class BugLabWarning(Enum):
    NO_WARNING = auto()
    INCONSISTENT_WARNING = auto()
    CONSISTENT_WARNING = auto()
    ANALYSIS_FAILED = auto()


def get_data_iterator_from(code_path: str, venv_location: Optional[str]):
    jedi_env = jedi.create_environment(venv_location) if venv_location else None

    try:
        with detect_encoding_and_open(code_path) as f:
            rel_db = PythonCodeRelations(f.read(), Path(code_path))
    except UnicodeDecodeError:
        return
    compute_all_relations(rel_db, jedi_env)

    available_ops: List[AbstractRewriteOp] = []
    available_ops_metadata: List[Tuple[Type[ICodeRewriteScout], cst.CSTNode, Any]] = []
    rel_db.ast_with_metadata_wrapper.visit_batched(
        [ScoutClass(available_ops, available_ops_metadata) for ScoutClass in ALL_REWRITE_SCOUTS]
    )

    function_finder = AllFunctionFinder()
    rel_db.ast_with_metadata_wrapper.visit(function_finder)
    for fn_node, fn_range in function_finder.all_function_nodes:
        relevant_ops, relevant_op_metadata = filter_ops_in_range(available_ops, available_ops_metadata, fn_range)
        yield get_serialized_representation(
            rel_db,
            fn_range,
            relevant_ops,
            relevant_op_metadata,
            target_fix_op=None,
            package_name="unknown package",
            package_version="unknown",
        ), relevant_ops


def get_fn_in_range(filepath, venv_location, fn_range):
    ranges = []
    for rewritten_fn_data, relevant_ops in get_data_iterator_from(filepath, venv_location):
        if rewritten_fn_data["graph"]["code_range"][0][0] == fn_range[0][0]:
            return rewritten_fn_data
        ranges.append(rewritten_fn_data["graph"]["code_range"])
    raise Exception("Target range not found")


def detect_bug(
    model,
    nn,
    device,
    fn_data,
    relevant_ops,
    filepath: str,
    venv_location,
    confidence_threshold: float,
    as_hypergraphs: bool,
) -> Tuple[BugLabWarning, Optional[AbstractRewriteOp]]:
    if as_hypergraphs:
        fn_data = convert_buglab_sample_to_hypergraph(fn_data)
    # Do we detect any bugs beyond the threshold? What is the suggested fix?
    predictions = model.predict([fn_data], nn, device, parallelize=False)
    try:
        datapoint, location_logprobs, rewrite_probs = next(predictions)
    except StopIteration:
        print(f'Sample could not be loaded from {fn_data["graph"]["code_range"]} in {filepath}.')
        return BugLabWarning.ANALYSIS_FAILED, None

    predicted_node_idx = max(location_logprobs, key=lambda k: location_logprobs[k])
    prediction_logprob = location_logprobs[predicted_node_idx]
    if predicted_node_idx == -1:
        return BugLabWarning.NO_WARNING, None

    predicted_rewrite_idx = None
    predicted_rewrite_logprob = -math.inf
    for rewrite_idx, (rewrite_node_idx, rewrite_logprob) in enumerate(
        zip(datapoint["graph"]["reference_nodes"], rewrite_probs)
    ):
        if rewrite_node_idx == predicted_node_idx and rewrite_logprob > predicted_rewrite_logprob:
            predicted_rewrite_idx = rewrite_idx
            predicted_rewrite_logprob = rewrite_logprob

    assert predicted_rewrite_idx is not None

    if np.exp(prediction_logprob + predicted_rewrite_logprob) < confidence_threshold:
        return BugLabWarning.NO_WARNING, None

    op = relevant_ops[predicted_rewrite_idx]
    assert fn_data["candidate_rewrites"][predicted_rewrite_idx] == (op.op_name(), op.rewrite_data())
    assert fn_data["candidate_rewrite_ranges"][predicted_rewrite_idx] == (
        (op.target_range.start.line, op.target_range.start.column),
        (op.target_range.end.line, op.target_range.end.column),
    )

    try:
        with detect_encoding_and_open(filepath) as f:
            original_code = f.read()
        # Apply bug fix
        rewritten, _ = op.rewrite(original_code)
        with open(filepath, "w") as f:
            f.write(rewritten)

        # does this change the prediction to NO_BUG?
        rewritten_fn_data = get_fn_in_range(filepath, venv_location, fn_data["graph"]["code_range"])
        if as_hypergraphs:
            rewritten_fn_data = convert_buglab_sample_to_hypergraph(rewritten_fn_data)
        predictions = model.predict([rewritten_fn_data], nn, device, parallelize=False)
        try:
            datapoint, location_logprobs, rewrite_probs = next(predictions)
        except StopIteration:
            print(f'Sample could not be loaded from {rewritten_fn_data["graph"]["code_range"]} in {filepath}.')
            return BugLabWarning.ANALYSIS_FAILED, None

        predicted_node_idx = max(location_logprobs, key=lambda k: location_logprobs[k])
        if predicted_node_idx == -1:
            return BugLabWarning.CONSISTENT_WARNING, op

        return BugLabWarning.INCONSISTENT_WARNING, op
    except FileNotFoundError as e:
        traceback.print_exc()
        return BugLabWarning.ANALYSIS_FAILED, None
    finally:
        # Revert rewrite
        with open(filepath, "w") as f:
            f.write(original_code)


def run(arguments):
    if arguments.venv_location is not None:
        venv_location = Path(arguments.venv_location)
        assert venv_location.exists()
    else:
        venv_location = None

    package_location = Path(arguments.package_path)
    assert package_location.exists()

    code_location = Path(arguments.code_dir)
    assert code_location.exists()

    confidence_threshold = arguments.confidence_level
    assert 0 <= confidence_threshold < 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = Path(arguments.model_path)
    model, nn = GnnBugLabModel.restore_model(model_path, device)

    num_files, num_functions, num_warnings_raised, num_warnings_unfiltered = 0, 0, 0, 0

    # For each location,
    for filepath in tqdm(code_location.glob("**/*.py")):
        num_files += 1
        for fn_data, relevant_ops in get_data_iterator_from(filepath, venv_location):
            num_functions += 1
            analysis_result, op = detect_bug(
                model,
                nn,
                device,
                fn_data,
                relevant_ops,
                filepath,
                venv_location,
                confidence_threshold,
                as_hypergraphs=arguments.hypergraph,
            )

            if analysis_result == BugLabWarning.CONSISTENT_WARNING:
                num_warnings_raised += 1
                num_warnings_unfiltered += 1
            elif analysis_result == BugLabWarning.INCONSISTENT_WARNING:
                num_warnings_raised += 1

    print(
        f"Num files: {num_files}, Num functions: {num_functions}, Num warnings raised: {num_warnings_raised}, Num warnings after filtering: {num_warnings_unfiltered} "
    )


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Filter warnings using internal consistency.")

    parser.add_argument("package_path", type=str, help="the directory where the package is located.")
    parser.add_argument("code_dir", type=str, help="the subdirectory within the package that contains the actual code.")

    parser.add_argument("model_path", type=str, help="the path to the BugLab model.")
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.1,
        help="The confidence threshold for deciding if a warning is to be tested.",
    )

    parser.add_argument(
        "--venv-location",
        type=str,
        help="The virtual environment location to run tests in.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug on exception.",
    )

    parser.add_argument(
        "--hypergraph",
        action="store_true",
        help="Model is hypergraph model",
    )

    args = parser.parse_args()
    run_and_debug(lambda: run(args), args.debug)
