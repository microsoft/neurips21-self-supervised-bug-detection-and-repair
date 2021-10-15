#!/usr/bin/env python
"""
Compute statistics about a dataset

Usage:
    stats.py [options] ALL_DATA_FOLDER

Options:
    -h --help                    Show this screen.
"""
from pathlib import Path

import numpy as np
from docopt import docopt

from buglab.utils.msgpackutils import load_msgpack_l_gz

if __name__ == "__main__":
    args = docopt(__doc__)
    dataset_dir = Path(args["ALL_DATA_FOLDER"])
    assert dataset_dir.exists()

    num_nodes_per_graph, num_edges_per_graph = [], []
    num_rewrite_locs, num_rewrite_ops = [], []

    for pkg_file in dataset_dir.rglob("*.msgpack.l.gz"):
        try:
            for graph in load_msgpack_l_gz(pkg_file):
                if graph is None:
                    continue
                num_nodes_per_graph.append(len(graph["graph"]["nodes"]))
                num_edges_per_graph.append(sum(len(e) for e in graph["graph"]["edges"].values()))
                num_rewrite_locs.append(len(np.unique(graph["graph"]["reference_nodes"])))
                num_rewrite_ops.append(len(graph["graph"]["reference_nodes"]))
        except Exception as e:
            print(f"Error loading {pkg_file}: {e} Skipping...")

    print(f"Num graphs: {len(num_nodes_per_graph)}")
    print(f"Nodes per graph: Avg:{np.mean(num_nodes_per_graph)} Median:{np.median(num_nodes_per_graph)}")
    print(f"Edges per graph: Avg:{np.mean(num_edges_per_graph)} Median:{np.median(num_edges_per_graph)}")
    print(f"Rewrite locations per graph: Avg:{np.mean(num_rewrite_locs)} Median: {np.median(num_rewrite_locs)}")
    print(f"Rewrite ops per graph: Avg:{np.mean(num_rewrite_ops)} Median: {np.median(num_rewrite_ops)}")
