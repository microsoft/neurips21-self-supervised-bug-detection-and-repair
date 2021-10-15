# !/usr/bin/env python
"""
Usage:
    asdot.py [options] SOURCE_CODE_FILE OUTPUT_DOT_FILE

Options:

    --filter-range=<VAL>       Comma-separated start,end line.
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
from pathlib import Path

from docopt import docopt
from dpu_utils.utils import run_and_debug
from libcst.metadata import CodeRange

from buglab.representations.coderelations import compute_all_relations
from buglab.representations.codereprs import PythonCodeRelations


def run(arguments):
    file = arguments["SOURCE_CODE_FILE"]
    target_dot_filepath = arguments["OUTPUT_DOT_FILE"]

    with open(file) as f:
        rel_db = PythonCodeRelations(f.read(), Path(file))
        compute_all_relations(rel_db)

        if arguments["--filter-range"] is not None:
            from_line, to_line = arguments["--filter-range"].split(",")
            target_range = CodeRange((int(from_line), 0), (int(to_line), 10000))
        else:
            target_range = None

        rel_db.as_dot(
            target_dot_filepath,
            {
                "LastMayWrite": "red",
                "NextMayUse": "blue",
                "OccurrenceOf": "green",
                "ComputedFrom": "brown",
                "NextToken": "hotpink",
                "Sibling": "lightgray",
            },
            target_range,
        )


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get("--debug", False))
