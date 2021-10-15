import unittest

import libcst as cst

from buglab.rewriting.semanticspreserving import (
    CommentDelete,
    IfElseSwap,
    MirrorComparisons,
    RenameVariableInLocalScope,
    _LocalVariablesFinder,
)
from tests.utils import iterate_buglab_code_files, token_equals


class TestSemanticsPreservingTransformations(unittest.TestCase):
    """
    Test semantics-preserving transformation on self.

    These tests simply test that the transformations are reversible, not the actual semantics preservation.

    """

    def test_if_else_swap(self):
        """
        Test if the transformation swapping if/else branches is reversible.

        This will fail in `not (x is y)` -> `x is not y`
        """
        for filepath in iterate_buglab_code_files():
            with self.subTest(f"Testing on {filepath}", path=filepath):
                with open(filepath) as f:
                    original = f.read()
                result = IfElseSwap.apply(original)
                result2 = IfElseSwap.apply(result)

                self.assertTrue(
                    token_equals(original, result2),
                    "Applying the reverse modification does not yield the original code.",
                )

    def test_mirror_comparisons(self):
        for filepath in iterate_buglab_code_files():
            with self.subTest(f"Testing on {filepath}", path=filepath):
                with open(filepath) as f:
                    original = f.read()
                result = MirrorComparisons.apply(original)
                result2 = MirrorComparisons.apply(result)

                self.assertTrue(
                    token_equals(original, result2),
                    "Applying the reverse modification does not yield the original code.",
                )

    def test_comment_delete(self):
        # Weak test: Make sure no exceptions are thrown
        for filepath in iterate_buglab_code_files():
            with self.subTest(f"Testing on {filepath}", path=filepath):
                with open(filepath) as f:
                    original = f.read()
                    r = CommentDelete.apply(original)

    def test_var_rename(self):
        # Weak test: Make sure no exceptions are thrown
        for filepath in iterate_buglab_code_files():
            with self.subTest(f"Testing on {filepath}", path=filepath):
                with open(filepath) as f:
                    original = f.read()
                r = RenameVariableInLocalScope.apply(original)


if __name__ == "__main__":
    unittest.main()
