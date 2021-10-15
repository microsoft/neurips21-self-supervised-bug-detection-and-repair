#!/usr/bin/env python
"""
Stable split of files based on filename hash

Usage:
    split.py [options] ALL_DATA_FOLDER OUTPUT_FOLDER

Options:
    -h --help                    Show this screen.
    --train-ratio FLOAT          Ratio of files for training set. [default: 0.6]
    --valid-ratio FLOAT          Ratio of files for validation set. [default: 0.2]
    --test-only-libraries=<File> A file containing the project names of the test-only data.
"""
import hashlib
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Set

from docopt import docopt

# Code adapted from https://github.com/microsoft/graph-based-code-modelling/blob/8967fac899628d74bf550d1ee2655c4cc85fa082/Models/utils/dataset_split.py
from buglab.utils.msgpackutils import load_msgpack_l_gz, save_msgpack_l_gz


def get_fold(filename: str, library: str, train_ratio: float, valid_ratio: float, test_only_libraries: Set[str]) -> str:
    if library in test_only_libraries:
        return "test-only"

    hash_val = int(hashlib.md5(filename.encode()).hexdigest(), 16) % (2 ** 16)
    train_bound = int(2 ** 16 * train_ratio)
    if hash_val <= train_bound:
        return "train"
    elif hash_val <= train_bound + int(2 ** 16 * valid_ratio):
        return "valid"
    else:
        return "test"


def split_file(
    input_path: Path,
    output_paths: Dict[str, Path],
    train_ratio: float,
    valid_ratio: float,
    test_only_libraries: Set[str],
) -> None:
    train_data, valid_data, test_data, test_only_data = [], [], [], []

    try:
        for datapoint in load_msgpack_l_gz(input_path):
            if datapoint is None:
                continue
            datapoint_provenance: str = datapoint["graph"]["path"]
            idx = datapoint_provenance.index("/venv/lib/python3.8/site-packages")
            assert idx >= 0
            datapoint_provenance = datapoint_provenance[idx + len("/venv/lib/python3.8/site-packages") :]
            library = datapoint["package_name"]
            file_set = get_fold(datapoint_provenance, library, train_ratio, valid_ratio, test_only_libraries)
            if file_set == "train":
                train_data.append(datapoint)
            elif file_set == "valid":
                valid_data.append(datapoint)
            elif file_set == "test":
                test_data.append(datapoint)
            elif file_set == "test-only":
                test_only_data.append(datapoint)
    except Exception as e:
        print(f"Failed for file {input_path}: {e}")
        return

    input_file_basename = input_path.name

    if len(train_data) > 0:
        output_path = output_paths["train"] / input_file_basename
        print("Saving %s..." % (output_path,))
        save_msgpack_l_gz(train_data, output_path)

    if len(valid_data) > 0:
        output_path = output_paths["valid"] / input_file_basename
        print("Saving %s..." % (output_path,))
        save_msgpack_l_gz(valid_data, output_path)

    if len(test_data) > 0:
        output_path = output_paths["test"] / input_file_basename
        print("Saving %s..." % (output_path,))
        save_msgpack_l_gz(test_data, output_path)

    if len(test_only_data) > 0:
        output_path = output_paths["test-only"] / input_file_basename
        print("Saving %s..." % (output_path,))
        save_msgpack_l_gz(test_only_data, output_path)


def split_many_files(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float,
    valid_ratio: float,
    test_only_libraries: Set[str],
    parallel: bool = True,
) -> None:
    output_paths = {}  # type: Dict[str, Path]
    for split_name in ["train", "valid", "test", "test-only"]:
        graph_dir_for_split_type = Path(str(output_dir) + "-" + split_name)
        output_paths[split_name] = graph_dir_for_split_type
        graph_dir_for_split_type.mkdir(exist_ok=True)

    if parallel:
        with Pool() as pool:
            pool.starmap(
                split_file,
                [
                    (f, output_paths, train_ratio, valid_ratio, test_only_libraries)
                    for f in input_dir.rglob("*.msgpack.l.gz")
                ],
            )
    else:
        for f in input_dir.rglob("*.msgpack.l.gz"):
            split_file(f, output_paths, train_ratio, valid_ratio, test_only_libraries)

    return None


if __name__ == "__main__":
    args = docopt(__doc__)
    train_ratio = float(args["--train-ratio"])
    valid_ratio = float(args["--valid-ratio"])
    test_ratio = 1 - train_ratio - valid_ratio
    assert 0 < train_ratio < 1, train_ratio
    assert 0 < valid_ratio < 1, valid_ratio
    assert 0 < test_ratio < 1, test_ratio

    test_only_libraries = set()  # type: Set[str]
    if args.get("--test-only-libraries") is not None:
        with open(args.get("--test-only-libraries")) as f:
            for line in f:
                if len(line.strip()) > 0:
                    test_only_libraries.add(line.strip())
    assert train_ratio + valid_ratio + test_ratio <= 1

    print(
        f"Splitting at the ratios {train_ratio:.2%}-{valid_ratio:.2%}-{test_ratio:.2%}. Test-only libraries: {test_only_libraries}"
    )

    data_folder = args["ALL_DATA_FOLDER"]
    if data_folder.endswith(
        "/"
    ):  # Split off final separator so that succeeding basename()s are not returning an empty string...
        data_folder = data_folder[:-1]
    data_folder = Path(data_folder)
    output_folder = Path(args["OUTPUT_FOLDER"])

    split_many_files(data_folder, output_folder, train_ratio, valid_ratio, test_only_libraries)

    print("Splitting finished.")
