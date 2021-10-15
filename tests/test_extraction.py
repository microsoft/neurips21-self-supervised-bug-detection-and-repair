import json
import unittest
from pathlib import Path
from threading import Thread
from typing import Any, List, Tuple, Type

import jedi
import libcst as cst
from jedi.api.environment import Environment

from buglab.controllers.buggydatacreation import extract_bugs_from
from buglab.controllers.helper.randombugselectorserver import random_bug_selector_server
from buglab.representations.coderelations import compute_all_relations
from buglab.representations.codereprs import PythonCodeRelations
from buglab.rewriting import ALL_REWRITE_SCOUTS, filter_ops_in_range
from buglab.rewriting.rewriteops import AbstractRewriteOp
from buglab.rewriting.rewritescouts import ICodeRewriteScout
from buglab.utils.cstutils import AllFunctionFinder
from tests.utils import get_all_files_for_package, iterate_buglab_test_snippets


class TestExtraction(unittest.TestCase):
    """Download a friendly package and run
    representation extraction on that package_name."""

    def _try_serialize_all(self, filepath: str, jedi_env: Environment):
        print(filepath)
        with open(filepath) as f:
            code_text = f.read()

        rel_db = PythonCodeRelations(code_text, Path(filepath))
        compute_all_relations(rel_db, jedi_env)

        function_finder = AllFunctionFinder()
        rel_db.ast_with_metadata_wrapper.visit(function_finder)

        available_ops: List[AbstractRewriteOp] = []
        available_ops_metadata: List[Tuple[Type[ICodeRewriteScout], cst.CSTNode, Any]] = []
        rel_db.ast_with_metadata_wrapper.visit_batched([ScoutClass(available_ops) for ScoutClass in ALL_REWRITE_SCOUTS])

        for function, fn_pos in function_finder.all_function_nodes:
            relevant_ops, relevant_op_metadata = filter_ops_in_range(available_ops, available_ops_metadata, fn_pos)
            serializable, _ = rel_db.as_serializable(
                target_range=fn_pos, reference_nodes=[node for _, node, _ in relevant_op_metadata]
            )
            _ = json.dumps(serializable)
            self.__assert_tokens_are_fully_connected(serializable["edges"]["NextToken"])

    def __assert_tokens_are_fully_connected(self, token_edges):
        next_token_edges = {f: t for f, t in token_edges}
        first_token = set(next_token_edges.keys()) - set(next_token_edges.values())
        self.assertEqual(len(first_token), 1, "The token sequence is disconnected.")

    def test_extraction_on_dpu_utils(self):
        for filepath, env in get_all_files_for_package("dpu-utils"):
            with self.subTest(f"Extracting on {filepath}", path=filepath):
                print(filepath)
                self._try_serialize_all(filepath, env)

    def test_extraction_on_azure_blob_storage(self):
        for filepath, env in get_all_files_for_package("azure-storage-blob"):
            with self.subTest(f"Extracting on {filepath}", path=filepath):
                print(filepath)
                self._try_serialize_all(filepath, env)

    def test_extraction_on_test_snippets(self):
        env = jedi.get_default_environment()
        for filepath in iterate_buglab_test_snippets():
            with self.subTest(f"Extracting on {filepath}", path=filepath):
                print(filepath)
                self._try_serialize_all(filepath, env)

    def test_random_extraction_on_test_snippets(self):
        env = jedi.get_default_environment()
        rewrite_selector_server_port: str = "8345"
        server_thread = Thread(
            target=lambda: random_bug_selector_server("tcp://*:" + rewrite_selector_server_port), daemon=True
        )
        server_thread.start()

        for filepath in iterate_buglab_test_snippets():
            with self.subTest(f"Extracting on {filepath}", path=filepath):
                all_p = extract_bugs_from(
                    filepath, env.path, "testsnippets", "0", "tcp://localhost:" + rewrite_selector_server_port
                )
                for extracted_fn in all_p:
                    for extracted_e, _ in extracted_fn["rewrites"].values():
                        self.__assert_tokens_are_fully_connected(extracted_e["graph"]["edges"]["NextToken"])


if __name__ == "__main__":
    unittest.main()
