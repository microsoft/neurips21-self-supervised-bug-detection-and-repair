#!/usr/bin/env python
"""
Extract the data from the concrete edits found during mining.

Usage:
    extract.py [options] DIFF_DATA BUGGY_OUTPUT_MSGPACK_L_GZ

Options:
    -h --help                    Show this screen.
    --debug                      Drop into pdb if there's an uncaught exception. [default: False]
"""
from typing import Any, Dict, List, Tuple, Type

import git
import json
import libcst as cst
import os
from collections import defaultdict
from docopt import docopt
from dpu_utils.utils import run_and_debug
from pathlib import Path
from tempfile import TemporaryDirectory

from buglab.controllers.buggydatacreation import get_serialized_representation
from buglab.representations.coderelations import compute_all_relations
from buglab.representations.codereprs import PythonCodeRelations
from buglab.rewriting import ALL_REWRITE_SCOUTS, AbstractRewriteOp, ICodeRewriteScout, filter_ops_in_range
from buglab.utils import detect_encoding_and_open
from buglab.utils.cstutils import AllFunctionFinder, subsumes_code_range
from buglab.utils.msgpackutils import save_msgpack_l_gz


def compute_relations(code_text, target_file):
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
    return available_ops, available_ops_metadata, function_finder, rel_db


def extract_from_file(target_file, target_rewrite: str, project_name, git_hash):
    with detect_encoding_and_open(target_file) as f:
        code_text = f.read()

    available_ops, available_ops_metadata, function_finder, rel_db = compute_relations(code_text, target_file)

    for function, fn_pos in function_finder.all_function_nodes:
        relevant_ops, relevant_op_metadata = filter_ops_in_range(available_ops, available_ops_metadata, fn_pos)
        for op in relevant_ops:
            if str(op) == target_rewrite:
                target_fix_op = op
                break
        else:
            continue

        buggy_code = get_serialized_representation(
            rel_db, fn_pos, relevant_ops, relevant_op_metadata, target_fix_op, project_name, git_hash
        )

        fixed_code_text, revert_op = target_fix_op.rewrite(code_text)
        fixed_available_ops, fixed_available_ops_metadata, fixed_function_finder, fixed_rel_db = compute_relations(
            fixed_code_text, target_file
        )
        for fixed_function, fixed_fn_pos in fixed_function_finder.all_function_nodes:
            fixed_relevant_ops, fixed_relevant_op_metadata = filter_ops_in_range(
                fixed_available_ops, fixed_available_ops_metadata, fixed_fn_pos
            )

            if subsumes_code_range(revert_op.target_range, fixed_fn_pos):
                fixed_code = get_serialized_representation(
                    fixed_rel_db,
                    fixed_fn_pos,
                    fixed_relevant_ops,
                    fixed_relevant_op_metadata,
                    None,
                    project_name,
                    git_hash,
                )
                break
        else:
            raise Exception("Fixed code operation not found.")

        return buggy_code, fixed_code
    else:
        pass  # For better or worse, these are probably outside of functions, which we have excluded so far.
    return None, None


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
                    buggy_code, fixed_code = extract_from_file(target_file_path, edit["rewrite"], repo_url, commit_hash)
                    if buggy_code is None:
                        continue
                    yield buggy_code, fixed_code
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
            except Exception as e:
                print(f"Error in extracting from {repo_url} because {e}")

    save_msgpack_l_gz(whole_dataset_extractor(), arguments["BUGGY_OUTPUT_MSGPACK_L_GZ"])


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get("--debug", False))
