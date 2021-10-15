import io
import json
import multiprocessing as mp
import traceback
from tokenize import COMMENT, ENCODING, NEWLINE, NL, TYPE_COMMENT, tokenize
from typing import Any, Iterator, List, Tuple, Type

import libcst as cst
from dpu_utils.utils import ThreadedIterator
from libcst import ParserSyntaxError
from libcst.metadata import CodeRange
from pydriller import ModificationType, RepositoryMining

from buglab.rewriting import ALL_REWRITE_SCOUTS, AbstractRewriteOp, ICodeRewriteScout, filter_ops_in_range
from buglab.utils.cstutils import PersistentMetadataWrapper


def _tokenize_no_comments(code: str) -> Iterator[str]:
    with io.BytesIO(code.encode()) as c:
        for toknum, tokval, _, _, _ in tokenize(c.readline):
            if tokval in {COMMENT, TYPE_COMMENT, NL, NEWLINE, ENCODING}:
                continue
            yield tokval


def token_equals_without_comments(code1: str, code2: str) -> bool:
    c1_tokens = list(_tokenize_no_comments(code1))
    c2_tokens = list(_tokenize_no_comments(code2))
    return c1_tokens == c2_tokens


def get_rewrite_ops(code, start_line, end_line):
    ast_with_wrapper = PersistentMetadataWrapper(cst.parse_module(code), unsafe_skip_copy=True)

    available_ops: List[AbstractRewriteOp] = []
    available_ops_metadata: List[Tuple[Type[ICodeRewriteScout], cst.CSTNode, Any]] = []

    ast_with_wrapper.visit_batched(
        [ScoutClass(available_ops, available_ops_metadata) for ScoutClass in ALL_REWRITE_SCOUTS]
    )
    relevant_ops, relevant_op_metadata = filter_ops_in_range(
        available_ops, available_ops_metadata, CodeRange((start_line, 0), (end_line, 999))
    )
    return relevant_ops, relevant_op_metadata


def get_simple_fixes(repo_url, max_size=5):
    miner = RepositoryMining(repo_url)
    for commit in miner.traverse_commits():
        if commit.merge:
            continue

        for modification in (m for m in commit.modifications if m.filename.endswith(".py")):
            if modification.change_type != ModificationType.MODIFY:
                continue
            if modification.added == 0 or modification.added > max_size:
                continue
            if modification.removed == 0 or modification.removed > max_size:
                continue
            added_lines = set(l[0] for l in modification.diff_parsed["added"])
            deleted_lines = set(l[0] for l in modification.diff_parsed["deleted"])
            if len(added_lines - deleted_lines) != 0 and len(deleted_lines - added_lines) != 0:
                continue
            start_line = min(min(added_lines), min(deleted_lines))
            end_line = max(max(added_lines), max(deleted_lines))

            if end_line - start_line > 6:
                continue

            try:
                relevant_ops, relevant_op_metadata = get_rewrite_ops(
                    modification.source_code_before, start_line, end_line
                )

                for rewrite_op, rewrite_op_metadata in zip(relevant_ops, relevant_op_metadata):
                    modified_code_text, reverse_op = rewrite_op.rewrite(modification.source_code_before)
                    if token_equals_without_comments(
                        modification.source_code, modified_code_text
                    ) and not token_equals_without_comments(modification.source_code_before, modified_code_text):
                        yield {
                            "repo": repo_url,
                            "message": commit.msg,
                            "hash": commit.hash,
                            "diff": modification.diff,
                            "old_path": modification.old_path,
                            "rewrite": str(rewrite_op),
                        }
                        break

            except ParserSyntaxError:
                pass
            except:
                traceback.print_exc()


def get_simple_fixes_as_list(repo_url):
    fixes = []
    try:
        for fix in get_simple_fixes(repo_url):
            fixes.append(fix)
    except:
        pass
    return fixes


def get_simple_fixes_from_repos(all_repos):
    with mp.Pool() as pool:
        results = pool.imap_unordered(get_simple_fixes_as_list, all_repos)
        for repo_results in results:
            yield from repo_results


if __name__ == "__main__":
    import sys

    repo_path = sys.argv[1]
    repos = set()
    with open(repo_path) as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            repos.add(line.strip())

    total_collected = 0
    for i, sample in enumerate(ThreadedIterator(get_simple_fixes_from_repos(repos), max_queue_size=10000)):
        print(f"------------------------ Sample {i}. Collected so far {total_collected} ---------------------------")
        print(">> ", sample["message"], ">>", sample["hash"], ">>", sample["repo"])
        print(sample["diff"])

        response = None
        while response not in ("y", "n"):
            response = input("Is valid? [y/n] ")

        if response == "y":
            total_collected += 1
            with open("bugs.jsonl", "a") as f:
                print(json.dumps(sample), file=f)
