#!/usr/bin/env python
"""
Filter dataset to only allow limited kinds of bugs.

Usage:
    filter_candidates_to_bugtype.py [options] INPUT_DATA_FOLDER OUTPUT_FOLDER

Options:
    -t --type [WRONG_OPERATOR|VARMISUSE]   Bug type to filter to. [default: WRONG_OPERATOR]
    -h --help                              Show this screen.
"""
from copy import deepcopy
from itertools import compress
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from docopt import docopt
from dpu_utils.utils import run_and_debug

from buglab.data.filter_data_to_cubert import (
    BugType,
    rewrite_is_cubert_varmisuse_bug,
    rewrite_is_cubert_wrong_operator_bug,
)
from buglab.utils.msgpackutils import load_msgpack_l_gz, save_msgpack_l_gz


def filter_datapoint_rewrites(
    datapoint: Dict[str, Any], rewrite_filter_fn: Callable[[Dict[str, Any], List[str], int], bool]
) -> Optional[Dict[str, Any]]:
    result = deepcopy(datapoint)

    # We compute the function_lines once, to avoid re-doing this for every filter step:
    function_text = datapoint["graph"]["text"]
    function_lines = function_text.split("\n")

    # Determine for each rewrite if it's appropriate:
    rewrite_mask = []
    for rewrite_idx in range(len(datapoint["candidate_rewrite_metadata"])):
        rewrite_mask.append(rewrite_filter_fn(datapoint, function_lines, rewrite_idx))

    # First, check if the target fix is actually allowed. If not, skip this example.
    target_fix_idx = datapoint["target_fix_action_idx"]
    if target_fix_idx is not None and not rewrite_mask[target_fix_idx]:
        return None

    # If present, we need to shift the index of the target fix by the number of things that we removed:
    if target_fix_idx is not None:
        offset = sum(1 if not m else 0 for m in rewrite_mask[:target_fix_idx])
        result["target_fix_action_idx"] = target_fix_idx - offset

    # Filter everything that's related to the rewrites:
    result["graph"]["reference_nodes"] = list(compress(datapoint["graph"]["reference_nodes"], rewrite_mask))
    result["candidate_rewrites"] = list(compress(datapoint["candidate_rewrites"], rewrite_mask))
    result["candidate_rewrite_ranges"] = list(compress(datapoint["candidate_rewrite_ranges"], rewrite_mask))
    result["candidate_rewrite_metadata"] = list(compress(datapoint["candidate_rewrite_metadata"], rewrite_mask))

    return result


def filter_file(
    input_path: Path,
    output_folder: Path,
    bug_type: BugType,
) -> None:
    output_path = output_folder.joinpath(input_path.name)

    if bug_type is BugType.WRONG_OPERATOR:
        rewrite_filter_fn = rewrite_is_cubert_wrong_operator_bug
    elif bug_type is BugType.VARMISUSE:
        rewrite_filter_fn = rewrite_is_cubert_varmisuse_bug
    else:
        raise ValueError(f"Filtering to bug type {bug_type} not supported.")

    filtered_datapoints = []
    num_read = 0
    for datapoint in load_msgpack_l_gz(input_path):
        if datapoint is None:
            continue
        num_read += 1
        filtered_datapoint = filter_datapoint_rewrites(datapoint, rewrite_filter_fn)
        if filtered_datapoint is not None:
            filtered_datapoints.append(filtered_datapoint)
    save_msgpack_l_gz(filtered_datapoints, output_path)
    print(f"Read {num_read} examples, filtered down to {len(filtered_datapoints)}.")
    print(f"Results stored in {output_path}.")


def run(args):
    bug_type = BugType[args["--type"]]
    out_folder = Path(args["OUTPUT_FOLDER"])
    out_folder.mkdir(parents=True, exist_ok=True)
    for pkg_file in Path(args["INPUT_DATA_FOLDER"]).rglob("*.msgpack.l.gz"):
        filter_file(pkg_file, out_folder, bug_type)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), True)
