from typing import Iterator, NamedTuple, Set

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


class PackageLabInfo(NamedTuple):
    venv_location: Path
    package_location: str
    package_name: str
    package_version: str
    all_package_files: Set[str]


def create_linux_venv(dir: Path, venv_location: Path, all_package_files: Path, package: str) -> int:
    command = f"""#!/bin/bash
    python3 -m venv {venv_location}
    source {venv_location / 'bin' / 'activate'}
    pip3 install --upgrade wheel pip
    pip3 install --no-cache-dir {package}
    pip3 show {package} --files > {all_package_files}
    """

    install_file = Path(dir) / "install.sh"
    with open(install_file, "w") as f:
        f.write(command)

    process = subprocess.run(f"bash {install_file}", shell=True)
    return process.returncode


def create_windows_venv(dir: Path, venv_location: Path, all_package_files: Path, package: str) -> int:
    command = f"""#!/usr/bin/pwsh
    python -m venv {venv_location}
    . {venv_location / 'Scripts' / 'Activate.ps1'}
    pip install --user --upgrade wheel pip
    pip install --no-cache-dir {package}
    pip show {package} --files | Out-File -FilePath {all_package_files}
    """

    install_file = Path(dir) / "install.ps1"
    with open(install_file, "w") as f:
        f.write(command)

    process = subprocess.run(f"powershell {install_file}", shell=True)
    return process.returncode


@contextmanager
def create_venv_and_install(package: str) -> Iterator[PackageLabInfo]:
    with TemporaryDirectory() as dir:
        dir = Path(dir)
        venv_location = dir / "venv"
        all_package_files = Path(dir) / "pkg-files.txt"

        if os.name == "nt":
            returncode = create_windows_venv(dir, venv_location, all_package_files, package)
        else:
            returncode = create_linux_venv(dir, venv_location, all_package_files, package)

        if returncode != 0:
            return None

        with open(all_package_files) as f:
            in_files_section = False
            all_files = set()
            for line in f.readlines():
                line = line.replace("\0", "")  # Remove null chars. For Windows compatibility.
                if line.startswith("Version: "):
                    pkg_version = line[len("Version: ") :].strip()
                elif line.startswith("Location: "):
                    pkg_location = line[len("Location: ") :].strip()
                elif line.startswith("Files:"):
                    in_files_section = True
                elif in_files_section:
                    all_files.add(line.strip())

        yield PackageLabInfo(venv_location, pkg_location, package, pkg_version, all_files)
