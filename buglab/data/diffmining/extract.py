#!/usr/bin/env python
"""
Extract the data from the concrete edits found during mining.

Usage:
    extract.py [options] DIFF_DATA OUTPUT_MSGPACK_L_GZ

Options:
    -h --help                    Show this screen.
    --debug                      Drop into pdb if there's an uncaught exception. [default: False]
"""
import json
import os
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple, Type

import git
import libcst as cst
from docopt import docopt
from dpu_utils.utils import run_and_debug

from buglab.controllers.buggydatacreation import get_serialized_representation
from buglab.representations.coderelations import compute_all_relations
from buglab.representations.codereprs import PythonCodeRelations
from buglab.rewriting import ALL_REWRITE_SCOUTS, AbstractRewriteOp, ICodeRewriteScout, filter_ops_in_range
from buglab.utils import detect_encoding_and_open
from buglab.utils.cstutils import AllFunctionFinder
from buglab.utils.msgpackutils import save_msgpack_l_gz


def extract_from_file(target_file, target_rewrite: str, project_name, git_hash):
    with detect_encoding_and_open(target_file) as f:
        code_text = f.read()

    # Jedi will do its best to find the project root
    rel_db = PythonCodeRelations(code_text, Path(target_file))
    compute_all_relations(rel_db, None)

    function_finder = AllFunctionFinder()
    rel_db.ast_with_metadata_wrapper.visit(function_finder)

    available_ops: List[AbstractRewriteOp] = []
    available_ops_metadata: List[Tuple[Type[ICodeRewriteScout], cst.CSTNode, Any]] = []
    rel_db.ast_with_metadata_wrapper.visit_batched(
        [ScoutClass(available_ops, available_ops_metadata) for ScoutClass in ALL_REWRITE_SCOUTS]
    )

    for function, fn_pos in function_finder.all_function_nodes:
        relevant_ops, relevant_op_metadata = filter_ops_in_range(available_ops, available_ops_metadata, fn_pos)

        for op in relevant_ops:
            if str(op) == target_rewrite:
                target_fix_op = op
                break
        else:
            continue

        return get_serialized_representation(
            rel_db, fn_pos, relevant_ops, relevant_op_metadata, target_fix_op, project_name, git_hash
        )
    else:
        pass  # For better or worse, these are probably outside of functions, which we have excluded so far.


def extract_from_repo(repo_url, diff_data: Dict[str, List]):
    with TemporaryDirectory() as tmp_dir:
        repo: git.Repo = git.Repo.clone_from(repo_url, tmp_dir)

        for commit_hash, elements_to_extract in diff_data.items():
            fix_commit: git.Commit = repo.commit(commit_hash)
            parents = fix_commit.parents
            if len(parents) > 1:
                # Avoid merges, for simplicity
                continue

            parent_commit: git.Commit = parents[0]
            repo.git.checkout(parent_commit)

            for edit in elements_to_extract:
                target_file_path = os.path.join(tmp_dir, edit["old_path"])
                try:
                    yield extract_from_file(target_file_path, edit["rewrite"], repo_url, commit_hash)
                except Exception as e:
                    print(f"Error extracting for {edit} because {e}")


def run(arguments):
    # Load diffs
    per_repo_diffs: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    with open(arguments["DIFF_DATA"]) as f:
        for line in f:
            line = json.loads(line)
            per_repo_diffs[line["repo"]][line["hash"]].append(line)

    def whole_dataset_extractor():
        for repo_url, repo_diff_data in per_repo_diffs.items():
            try:
                yield from extract_from_repo(repo_url, repo_diff_data)
            except:
                print(f"Error in extracting from {repo_url}")

    save_msgpack_l_gz(whole_dataset_extractor(), arguments["OUTPUT_MSGPACK_L_GZ"])


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get("--debug", False))
