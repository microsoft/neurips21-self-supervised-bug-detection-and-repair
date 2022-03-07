from typing import Any, List, Tuple, Type

try:
    from typing import Final
except ImportError:
    from typing_extensions import Final

import libcst as cst
from libcst.metadata import CodeRange

from buglab.rewriting.rewriteops import AbstractRewriteOp
from buglab.rewriting.rewritescouts import (
    ArgSwapRewriteScout,
    AssignRewriteScout,
    BinaryOperatorRewriteScout,
    BooleanOperatorRewriteScout,
    ComparisonOperatorRewriteScout,
    ICodeRewriteScout,
    LiteralRewriteScout,
    VariableMisuseRewriteScout,
)
from buglab.rewriting.semanticspreserving import (
    CommentDelete,
    IfElseSwap,
    MirrorComparisons,
    RemoveTypeAnnotations,
    RenameVariableInLocalScope,
    ShuffleLiteralConstructors,
    ShuffleNamedArgs,
)
from buglab.utils.cstutils import subsumes_code_range

ALL_REWRITE_SCOUTS: Final = (
    ArgSwapRewriteScout,
    AssignRewriteScout,
    BinaryOperatorRewriteScout,
    BooleanOperatorRewriteScout,
    ComparisonOperatorRewriteScout,
    LiteralRewriteScout,
    VariableMisuseRewriteScout,
)

ALL_SEMANTICS_PRESERVING_TRANSFORMS: Final = (
    MirrorComparisons,
    RenameVariableInLocalScope,
    CommentDelete,
    IfElseSwap,
    ShuffleNamedArgs,
    RemoveTypeAnnotations,
    ShuffleLiteralConstructors,
)


def filter_ops_in_range(
    available_ops: List[AbstractRewriteOp],
    available_ops_metadata: List[Tuple[Type[ICodeRewriteScout], cst.CSTNode, Any]],
    target_range: CodeRange,
) -> Tuple[List[AbstractRewriteOp], List[Tuple[Type[ICodeRewriteScout], cst.CSTNode, Any]]]:
    relevant_op_metadata: List[Tuple[Type[ICodeRewriteScout], cst.CSTNode, Any]] = []

    def get_ops_in_range():
        for op, op_metadata in zip(available_ops, available_ops_metadata):
            if subsumes_code_range(op.target_range, target_range):
                yield op
                relevant_op_metadata.append(op_metadata)

    return list(get_ops_in_range()), relevant_op_metadata


def apply_semantics_preserving_transforms(code: str, p_apply_each: float = 0.1) -> str:
    for transform in ALL_SEMANTICS_PRESERVING_TRANSFORMS:
        code = transform.apply(code, p_apply_each)
    return code
