#!/usr/bin/env python
"""
Create a subfolder of folder with graphs converted to hypergraphs. E.g. 'train' folder gets 'train/hypergraphs' folder

Usage:
    convert_data_to_hypergraph.py [options] (buglab|types) INPUT_DATA_FOLDER TARGET_DATA_FOLDER


Options:
    -h --help                    Show this screen.
    --chunk-size CS              NextToken might be large, so we split it into chunks of size `--chunk-size`. [default: 255]
    --sequential
"""
import multiprocessing
import os
from docopt import docopt
from functools import partial
from pathlib import Path

from buglab.representations.hypergraph import convert_buglab_sample_to_hypergraph, convert_type_sample_to_hypergraph
from buglab.utils.msgpackutils import load_msgpack_l_gz, save_msgpack_l_gz


def convert_file(input_file, target_folder, conversion_fn):
    print("pkg_file", input_file)
    _, filename = os.path.split(input_file)

    def convert_to_hypergraphs():
        try:
            for graph in load_msgpack_l_gz(input_file):
                if graph is None:
                    continue
                try:
                    assert len(graph["graph"]["edges"]) != 0
                    hypergraph = conversion_fn(graph, chunk_size=int(args["--chunk-size"]))
                    assert len(hypergraph["graph"]["edges"]) == 0
                    yield hypergraph
                except Exception as e:
                    print(f"Error processing graph in {input_file}, {e}.")

        except Exception as e:
            print(f"Error processing file {input_file}, {e}.")

    filename = os.path.join(target_folder, filename)
    save_msgpack_l_gz(convert_to_hypergraphs(), filename)


if __name__ == "__main__":
    args = docopt(__doc__)
    target_dir = args["TARGET_DATA_FOLDER"]
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args["INPUT_DATA_FOLDER"])
    # new_data_folder
    assert dataset_dir.exists()

    if args["types"]:
        conversion_fn = convert_type_sample_to_hypergraph
    elif args["buglab"]:
        conversion_fn = convert_buglab_sample_to_hypergraph
    else:
        raise Exception()

    if args["--sequential"]:
        for f in dataset_dir.glob("*.msgpack.l.gz"):
            convert_file(f, target_folder=target_dir, conversion_fn=conversion_fn)
    else:
        with multiprocessing.Pool() as pool:
            pool.map(
                partial(convert_file, target_folder=target_dir, conversion_fn=conversion_fn),
                dataset_dir.glob("*.msgpack.l.gz"),
            )
