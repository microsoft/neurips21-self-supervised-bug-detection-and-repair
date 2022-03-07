# !/usr/bin/env python
"""
Usage:
    asdot.py [options] SOURCE_CODE_FILE OUTPUT_DOT_FILE

Options:

    --filter-range=<VAL>       Comma-separated start,end line.
    --as-hypergraph
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
from copy import deepcopy
from docopt import docopt
from dpu_utils.utils import run_and_debug
from libcst.metadata import CodeRange
from pathlib import Path

from buglab.representations.coderelations import compute_all_relations
from buglab.representations.codereprs import PythonCodeRelations
from buglab.representations.data import BugLabGraph
from buglab.representations.hypergraph import convert_to_hypergraph


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
            target_range = CodeRange((0, 0), (100000000, 10000))

        if arguments["--as-hypergraph"]:
            data, _ = rel_db.as_serializable(target_range, [])
            nxt_token = deepcopy(data["edges"]["NextToken"])
            hgraph, _ = convert_to_hypergraph(data)

            # Include token sequence as a visualization aid
            hgraph["edges"]["NextToken"] = nxt_token
            BugLabGraph.to_dot(hgraph, target_dot_filepath, {})

        else:
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
