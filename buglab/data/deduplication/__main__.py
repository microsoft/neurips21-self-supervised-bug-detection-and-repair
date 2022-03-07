import argparse
import gzip
import logging
import pickle
from pathlib import Path

from buglab.utils.loggingutils import configure_logging

from .index import DuplicationIndex

if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="A server process for keeping track of duplicates.")
    parser.add_argument("save_path", type=str, help="a location to the state of the server.")
    parser.add_argument("--address", default="tcp://*:5555", help="Address where the server will be listening at.")

    args = parser.parse_args()
    save_path = Path(args.save_path)
    if save_path.exists():
        logging.info("Restoring Duplication Index from %s", save_path)
        with gzip.open(save_path) as f:
            duplication_index = pickle.load(f)
    else:
        logging.info("New Duplication Index created.")
        duplication_index = DuplicationIndex(save_path)

    duplication_index.server(address=args.address)
