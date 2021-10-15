import argparse
import json
import math
import os
import subprocess
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List, Optional, Tuple, Type

import jedi
import libcst as cst
import numpy as np
import torch
from dpu_utils.utils import run_and_debug
from tqdm import tqdm

from buglab.controllers.buggydatacreation import get_serialized_representation
from buglab.models.gnn import GnnBugLabModel
from buglab.representations.coderelations import compute_all_relations
from buglab.representations.codereprs import PythonCodeRelations
from buglab.rewriting import ALL_REWRITE_SCOUTS, AbstractRewriteOp, ICodeRewriteScout, filter_ops_in_range
from buglab.utils import detect_encoding_and_open
from buglab.utils.cstutils import AllFunctionFinder
from buglab.utils.logging import configure_logging


def exec_command(venv_location: Optional[Path], command: str):
    print(f"Executing `{command}`")
    if venv_location is None:
        command = f"""#!/bin/bash
                {command}
                """
    else:
        command = f"""#!/bin/bash
        source {venv_location / 'bin' / 'activate'}
        {command}
        """

    with TemporaryDirectory() as tmp:
        command_file = Path(tmp) / "run.sh"
        with open(command_file, "w") as f:
            f.write(command)

        process = subprocess.run(f"bash {command_file}", shell=True)
        return process.returncode


def get_data_iterator_from(
    code_path: str, venv_location: Optional[str], covered_lines: List[int], report_uncovered: bool
):
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
        if not report_uncovered and not any(fn_range.start.line <= l <= fn_range.end.line for l in covered_lines):
            # Not covered.
            continue

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

    #  Run unit tests with coverage
    #  Requires `pip install pytest-json-report`
    exec_command(
        venv_location,
        f"coverage run --branch --source {code_location.absolute()} -m pytest {package_location.absolute()} --json-report --json-report-summary",
    )
    exec_command(venv_location, "coverage json")
    exec_command(venv_location, "coverage report")
    # TODO: In a future version, the coverage needs to be run per-test. Then only the subset of tests that
    #  cover a particular location need to be executed instead of the whole test suite.

    with open(".report.json") as f:
        base_test_report = json.load(f)["summary"]
    with open("coverage.json") as f:
        test_coverage = json.load(f)["files"]
    os.unlink(".report.json")

    test_coverage = {f: d for f, d in test_coverage.items() if str(code_location) in str(Path(f).absolute())}
    print(f"{len(test_coverage)} files covered by the tests")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = Path(arguments.model_path)
    model, nn = GnnBugLabModel.restore_model(model_path, device)

    num_files, num_functions, num_warnings_raised, num_warnings_unfiltered = 0, 0, 0, 0
    # For each location,
    for filepath, coverage_data in tqdm(test_coverage.items()):
        num_files += 1
        for covered_fn_data, relevant_ops in get_data_iterator_from(
            filepath, venv_location, coverage_data["executed_lines"], arguments.report_uncovered
        ):
            num_functions += 1
            # Do we detect any bugs beyond the threshold? What is the suggested fix?
            predictions = model.predict([covered_fn_data], nn, device, parallelize=False)
            try:
                datapoint, location_logprobs, rewrite_probs = next(predictions)
            except StopIteration:
                print(
                    f'Sample could not be loaded from Testing at range {covered_fn_data["graph"]["code_range"]} in {filepath}.'
                )
                continue

            predicted_node_idx = max(location_logprobs, key=lambda k: location_logprobs[k])
            prediction_logprob = location_logprobs[predicted_node_idx]
            if predicted_node_idx == -1:
                continue
            num_warnings_raised += 1

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
                continue

            op = relevant_ops[predicted_rewrite_idx]
            assert covered_fn_data["candidate_rewrites"][predicted_rewrite_idx] == (op.op_name(), op.rewrite_data())
            assert covered_fn_data["candidate_rewrite_ranges"][predicted_rewrite_idx] == (
                (op.target_range.start.line, op.target_range.start.column),
                (op.target_range.end.line, op.target_range.end.column),
            )

            if not any(
                op.target_range.start.line <= l <= op.target_range.end.line for l in coverage_data["executed_lines"]
            ):
                if arguments.report_uncovered:
                    num_warnings_unfiltered += 1
                    print(f'Testing at range {covered_fn_data["graph"]["code_range"]} in {filepath}...')
                    print(f"\tApplying candidate bug fix {op}")
                    print("\tTest Suite does not cover warning. Potential bug:")
                    print(f"\t Predicted rewrite: {op}")
                    print("\n")
                continue

            try:
                with detect_encoding_and_open(filepath) as f:
                    original_code = f.read()
                # Apply bug fix
                rewritten, _ = op.rewrite(original_code)
                with open(filepath, "w") as f:
                    f.write(rewritten)

                # does this change they pytest results? if not, report candidate.
                exec_command(
                    venv_location,
                    f"pytest {package_location.absolute()} --json-report --json-report-summary > /dev/null",
                )
                with open(".report.json") as f:
                    after_fix_test_report = json.load(f)["summary"]
                os.unlink(".report.json")

                if base_test_report != after_fix_test_report:
                    num_warnings_unfiltered += 1
                    print(f'Testing at range {covered_fn_data["graph"]["code_range"]} in {filepath}...')
                    print(f"\tApplying candidate bug fix {op}")
                    print("\tTest Suite could not filter bug fix suggestions. Potential bug:")
                    print(f"\t Predicted rewrite: {op}")
                    print("\n")
            except FileNotFoundError as e:
                traceback.print_exc()
            finally:
                # Revert bug fix
                with open(filepath, "w") as f:
                    f.write(original_code)
    print(
        f"Num files: {num_files}, Num functions: {num_functions}, Num warnings raised: {num_warnings_raised}, Num warnings undetected: {num_warnings_unfiltered} "
    )


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Filter warnings using unit tests.")

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
        "--report-uncovered",
        action="store_true",
        help="Report raised warnigns when they are uncovered by tests.",
    )

    args = parser.parse_args()
    run_and_debug(lambda: run(args), args.debug)
