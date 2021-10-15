import gzip
import random
from collections import OrderedDict
from os import PathLike
from typing import Any, Iterable, Iterator, Optional

import msgpack
from dpu_utils.utils import RichPath


def load_msgpack_l_gz(filename: PathLike) -> Iterator[Any]:
    with gzip.open(filename) as f:
        unpacker = msgpack.Unpacker(f, raw=False, object_pairs_hook=OrderedDict)
        yield from unpacker


def save_msgpack_l_gz(data: Iterable[Any], filename: PathLike) -> None:
    with gzip.GzipFile(filename, "wb") as out_file:
        packer = msgpack.Packer(use_bin_type=True)
        for element in data:
            out_file.write(packer.pack(element))


def load_all_msgpack_l_gz(
    path: RichPath,
    shuffle: bool = False,
    take_only_first_n_files: Optional[int] = None,
    limit_num_yielded_elements: Optional[int] = None,
) -> Iterator:
    all_files = sorted(path.iterate_filtered_files_in_dir("*.msgpack.l.gz"))
    if take_only_first_n_files is not None:
        all_files = all_files[:take_only_first_n_files]
    if shuffle:
        random.shuffle(all_files)

    sample_idx = 0
    for msgpack_file in all_files:
        try:
            for element in load_msgpack_l_gz(msgpack_file.to_local_path().path):
                if element is not None:
                    sample_idx += 1
                    yield element
                if limit_num_yielded_elements is not None and sample_idx > limit_num_yielded_elements:
                    return
        except Exception as e:
            print(f"Error loading {msgpack_file}: {e}.")


if __name__ == "__main__":
    # A json.tool-like CLI
    import json
    import sys

    for datapoint in load_msgpack_l_gz(sys.argv[1]):
        print(json.dumps(datapoint, indent=2))
        print("---------------------------------------")
