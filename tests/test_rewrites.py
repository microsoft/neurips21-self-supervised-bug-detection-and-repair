import unittest
from typing import Type

import libcst as cst
from tqdm import tqdm

from buglab.rewriting import ALL_REWRITE_SCOUTS
from buglab.rewriting.rewriteops import AbstractRewriteOp
from buglab.rewriting.rewritescouts import ICodeRewriteScout
from tests.utils import get_all_files_for_package, iterate_buglab_code_files, iterate_buglab_test_snippets


class TestCodeRewrites(unittest.TestCase):
    def test_roundtrip_on_self(self):
        for filepath in iterate_buglab_code_files():
            with self.subTest(f"Testing on {filepath}", path=filepath):
                print(filepath)
                self.assert_round_trip(filepath)

    def test_roundtrip_on_test_snippets(self):
        for filepath in iterate_buglab_test_snippets():
            with self.subTest(f"Testing on {filepath}", path=filepath):
                print(filepath)
                self.assert_round_trip(filepath)

    def test_roundtrip_on_azure_blob(self):
        for filepath, env in get_all_files_for_package("azure-storage-blob"):
            with self.subTest(f"Testing on {filepath}", path=filepath):
                print(filepath)
                self.assert_round_trip(filepath)

    def test_roundtrip_on_dpu_utils(self):
        for filepath, env in get_all_files_for_package("dpu-utils"):
            with self.subTest(f"Testing on {filepath}", path=filepath):
                print(filepath)
                self.assert_round_trip(filepath)

    def assert_round_trip(self, path: str):
        self.maxDiff = None
        with open(path) as f:
            code_text = f.read()
            wrapper = cst.MetadataWrapper(module=cst.parse_module(code_text), unsafe_skip_copy=True)

            available_ops = []
            wrapper.visit_batched([RetrieverClass(available_ops) for RetrieverClass in ALL_REWRITE_SCOUTS])
            for op in tqdm(available_ops, desc="Check Roundtrip", leave=False):
                modified, reverse_op = op.rewrite(code_text)
                original, forward_op = reverse_op.rewrite(modified)
                self.assertEqual(
                    original, code_text, "Applying the reverse modification does not yield the original code."
                )
                self.assertEqual(op.target_range, forward_op.target_range)
                self.assertEqual(op.op_name(), forward_op.op_name())
                modified_ast = cst.parse_module(modified)  # Check that modified code parses without errors
                self.assertIsNotNone(modified_ast)

    def test_reversible_on_self(self):
        """Test that rewrite scouts find rewrites that the scout can revert in a second round."""
        for filepath in iterate_buglab_code_files():
            with self.subTest(f"Testing on {filepath}", path=filepath):
                print(filepath)
                self.assert_all_ops_reversible(filepath)

    def test_reversible_on_test_snippets(self):
        for filepath in iterate_buglab_test_snippets():
            with self.subTest(f"Testing on {filepath}", path=filepath):
                print(filepath)
                self.assert_all_ops_reversible(filepath)

    def test_reversible_on_azure_blob(self):
        """Test that rewrite scouts find rewrites that the scout can revert in a second round."""
        for filepath, env in get_all_files_for_package("azure-storage-blob"):
            with self.subTest(f"Testing on {filepath}", path=filepath):
                print(filepath)
                self.assert_all_ops_reversible(filepath)

    def test_reversible_on_dpu_utils(self):
        """Test that rewrite scouts find rewrites that the scout can revert in a second round."""
        for filepath, env in get_all_files_for_package("dpu-utils"):
            with self.subTest(f"Testing on {filepath}", target_path=filepath):
                print(filepath)
                self.assert_all_ops_reversible(filepath)

    def assert_all_ops_reversible(self, code_path):
        with open(code_path) as f:
            code_text = f.read()
        wrapper = cst.MetadataWrapper(cst.parse_module(code_text), unsafe_skip_copy=True)
        for RewriteScoutClass in ALL_REWRITE_SCOUTS:
            available_ops = []
            wrapper.visit_batched([RewriteScoutClass(available_ops)])

            for op in tqdm(available_ops, desc=f"Check Reversible Ops from {RewriteScoutClass.__name__}", leave=False):
                modified, reverse_op = op.rewrite(code_text)
                self._assert_has_reversible_op(modified, RewriteScoutClass, reverse_op)

    def _assert_has_reversible_op(
        self, modified_code: str, RewriteScoutClass: Type[ICodeRewriteScout], reverse_op: AbstractRewriteOp
    ):
        wrapper = cst.MetadataWrapper(module=cst.parse_module(modified_code), unsafe_skip_copy=True)

        available_ops = []
        wrapper.visit_batched([RewriteScoutClass(available_ops)])
        self.assertIn(reverse_op, available_ops)


if __name__ == "__main__":
    unittest.main()
