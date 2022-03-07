from typing import List

import argparse
import logging
import os
import subprocess
from concurrent import futures
from enum import Enum
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Thread

from buglab.controllers.helper.randombugselectorserver import random_bug_selector_server
from buglab.data.deduplication import DuplicationIndex
from buglab.utils.loggingutils import configure_logging

LOGGER = logging.getLogger(__name__)


class ExtractJob(Enum):
    BUG = 1
    TYPE = 2


def create_container_and_extract(
    package_name: str, target_dir: PathLike, bug_selector_server_address: str, extract_job: ExtractJob
):
    docker_command = f'docker run --network="host" --rm -it -v {target_dir}:/data/targetDir buglab-base:latest '

    if extract_job == ExtractJob.BUG:
        docker_command += (
            f"python3.8 -m buglab.controllers.packageextracttodisk {package_name} "
            f"--bug-selector-server {bug_selector_server_address}"
        )
    elif extract_job == ExtractJob.TYPE:
        docker_command += f"python3.8 -m buglab.controllers.typelessextracttodisk {package_name} "
    else:
        raise ValueError(f"Unknown extraction job type: {extract_job}")

    _ = subprocess.run(docker_command, shell=True)


def extract_from_packages(
    packages: List[str], target_dir: PathLike, bug_selector_server_address: str, extract_job: ExtractJob
):
    with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        all_futures = {
            executor.submit(
                create_container_and_extract, pkg, target_dir, bug_selector_server_address, extract_job
            ): pkg
            for pkg in packages
        }
        for future in futures.as_completed(all_futures):
            try:
                _ = future.result()
            except Exception as exc:
                LOGGER.exception("Failure for `%s` package", all_futures[future], exc_info=exc)


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Orchestrator to extract graphs across multiple packages.")
    parser.add_argument(
        "package_list_path",
        type=str,
        help="the path to a txt file containing the names of the packages to be considered",
    )
    parser.add_argument("target_dir", type=str, help="the target directory to store the results.")
    parser.add_argument(
        "--extract-types",
        action="store_true",
        help="Set this flag if you want to remove type information instead of extracting bugs.",
    )
    args = parser.parse_args()

    f = NamedTemporaryFile()
    duplication_index = DuplicationIndex(Path(f.name))
    duplication_server_thread = Thread(target=lambda: duplication_index.server(address="tcp://*:5555"), daemon=True)
    duplication_server_thread.start()

    rewrite_selector_server_port: str = "8345"
    extract_job = ExtractJob.TYPE
    if not args.extract_types:
        extract_job = ExtractJob.BUG
        server_thread = Thread(
            target=lambda: random_bug_selector_server("tcp://*:" + rewrite_selector_server_port), daemon=True
        )
        server_thread.start()

    all_packages = []
    with open(args.package_list_path) as f:
        for line in f.readlines():
            pkg_name = line.strip()
            if len(pkg_name) > 0:
                all_packages.append(pkg_name)
    extract_from_packages(all_packages, args.target_dir, "tcp://localhost:" + rewrite_selector_server_port, extract_job)
