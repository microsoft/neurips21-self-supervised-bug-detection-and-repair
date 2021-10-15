#!/usr/bin/env python
"""
Translate and filter dataset of our bugs into the CuBERT input format.

Usage:
    filter_data_to_cubert.py [options] INPUT_DATA_FOLDER OUTPUT_FOLDER

Options:
    -t --type [CORRECT|WRONG_OPERATOR|VARMISUSE]   Bug type to filter to. [default: CORRECT]
    -h --help                                      Show this screen.
"""
import enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from docopt import docopt
from dpu_utils.utils import run_and_debug

from buglab.utils.msgpackutils import load_msgpack_l_gz


class BugType(enum.Enum):
    CORRECT = 0
    WRONG_OPERATOR = 1
    VARMISUSE = 2


# Taken from Table 7 of CuBERT paper:
CUBERT_ALLOWED_OPERATORS = {
    "+",
    "*",
    "-",
    "/",
    "%",
    "==",
    "!=",
    "is",
    "is not",
    "<",
    "<=",
    ">",
    ">=",
    "in",
    "not in",
    "and",
    "or",
}


def rewrite_is_cubert_wrong_operator_bug(
    datapoint: Dict[str, Any], function_lines: List[str], rewrite_idx: Optional[int]
) -> bool:
    if rewrite_idx is None:
        return False

    picked_rewrite_metadata = datapoint["candidate_rewrite_metadata"][rewrite_idx]
    picked_rewrite_type = picked_rewrite_metadata[0]

    if picked_rewrite_type not in (
        "ComparisonOperatorRewriteScout",
        "BooleanOperatorRewriteScout",
        "BinaryOperatorRewriteScout",
    ):
        return False

    rewritten_range = datapoint["candidate_rewrite_ranges"][rewrite_idx]
    # We don't do multiline splitting right now:
    if rewritten_range[0][0] != rewritten_range[1][0]:
        return False
    # We need to translate line indices into our range to get the old operator:
    text_start_line = datapoint["graph"]["code_range"][0][0]
    fun_rewrite_line = rewritten_range[0][0] - text_start_line
    old_operator = function_lines[fun_rewrite_line][rewritten_range[0][1] : rewritten_range[1][1]].strip()
    new_operator = datapoint["candidate_rewrites"][rewrite_idx][1].strip()

    if old_operator not in CUBERT_ALLOWED_OPERATORS or new_operator not in CUBERT_ALLOWED_OPERATORS:
        return False

    return True


def rewrite_is_cubert_varmisuse_bug(
    datapoint: Dict[str, Any], function_lines: List[str], rewrite_idx: Optional[int]
) -> bool:
    if rewrite_idx is None:
        return False

    picked_rewrite_metadata = datapoint["candidate_rewrite_metadata"][rewrite_idx]
    picked_rewrite_type = picked_rewrite_metadata[0]

    if picked_rewrite_type not in ("VariableMisuseRewriteScout",):
        return False

    # TODO: Do we need to do any more filtering?

    return True


def translate_correct_example(datapoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    picked_rewrite_idx = datapoint["target_fix_action_idx"]
    if picked_rewrite_idx is not None:
        return None

    function_text = datapoint["graph"]["text"]
    return {
        "function": function_text,
        "label": "Correct",
    }


def translate_operator_example(datapoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    function_text = datapoint["graph"]["text"]
    function_lines = function_text.split("\n")
    picked_rewrite_idx = datapoint["target_fix_action_idx"]

    if not rewrite_is_cubert_wrong_operator_bug(datapoint, function_lines, picked_rewrite_idx):
        return None

    return {
        "function": function_text,
        "label": "Wrong binary operator",
    }


def translate_varmisuse_example(datapoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    function_text = datapoint["graph"]["text"]
    function_lines = function_text.split("\n")
    picked_rewrite_idx = datapoint["target_fix_action_idx"]

    if not rewrite_is_cubert_varmisuse_bug(datapoint, function_lines, picked_rewrite_idx):
        return None

    return {
        "function": function_text,
        "label": "Variable misuse",
    }


def translate_file(
    input_path: Path,
    output_folder: Path,
    bug_type: BugType,
) -> None:
    filename = input_path.name + ".jsonl"
    output_path = output_folder.joinpath(filename)
    with open(output_path, "wt") as out_fh:
        num_read, num_written = 0, 0
        for datapoint in load_msgpack_l_gz(input_path):
            if datapoint is None:
                continue
            num_read += 1

            if bug_type is BugType.CORRECT:
                translated_sample = translate_correct_example(datapoint)
            elif bug_type is BugType.WRONG_OPERATOR:
                translated_sample = translate_operator_example(datapoint)
            elif bug_type is BugType.VARMISUSE:
                translated_sample = translate_varmisuse_example(datapoint)

            if translated_sample is None:
                continue

            out_fh.write(json.dumps(translated_sample) + "\n")
            num_written += 1
    print(f"Read {num_read} examples, filtered down to {num_written}.")
    print(f"Results stored in {output_path}.")


def run(args):
    bug_type = BugType[args["--type"]]
    out_folder = Path(args["OUTPUT_FOLDER"])
    out_folder.mkdir(parents=True, exist_ok=True)
    for pkg_file in Path(args["INPUT_DATA_FOLDER"]).rglob("*.msgpack.l.gz"):
        translate_file(pkg_file, out_folder, bug_type)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), True)
