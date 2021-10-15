import io
import os
from glob import iglob
from itertools import zip_longest
from pathlib import Path
from tokenize import tokenize
from typing import Iterator

import jedi

from buglab.data.pypi.venv import create_venv_and_install


def iterate_buglab_code_files() -> Iterator[str]:
    base_path = Path(__file__).parent.parent
    for python_file in iglob(os.path.join(str(base_path), "**", "*.py"), recursive=True):
        yield python_file


def iterate_buglab_test_snippets() -> Iterator[str]:
    base_path = Path(__file__).parent / "testsnippets"
    for python_file in iglob(os.path.join(str(base_path), "**", "*.py"), recursive=True):
        yield python_file


def token_equals(code1: str, code2: str) -> bool:
    with io.BytesIO(code1.encode()) as c1, io.BytesIO(code2.encode()) as c2:
        tokens1_iter = tokenize(c1.readline)
        tokens2_iter = tokenize(c2.readline)
        for (
            (toknum1, tokval1, _, _, _),
            (toknum2, tokval2, _, _, _),
        ) in zip_longest(tokens1_iter, tokens2_iter, fillvalue=(None, None, 0, 0, 0)):
            if toknum1 != toknum2 or tokval1 != tokval2:
                return False
        return True


def get_all_files_for_package(package_name: str):
    with create_venv_and_install(package_name) as created_env:
        jedi_env = jedi.create_environment(created_env.venv_location)

        for file in created_env.all_package_files:
            if not file.endswith(".py"):
                continue
            yield os.path.join(created_env.package_location, file), jedi_env
