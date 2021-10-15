from collections import defaultdict
from random import Random
from typing import Dict, Final, Optional, Set, Union

import libcst as cst
from libcst.metadata import QualifiedName, QualifiedNameProvider, QualifiedNameSource, Scope, ScopeProvider


class MirrorComparisons(cst.CSTTransformer):
    """
    Mirror comparison operands. This is _mostly_ semantics preserving, with the exception of the multiple comparison targets.

    For example:
        foo(1) < foo(0) is transformed to foo(0) > foo(1)

    If strict=True then comparisons with multiple comparison targets are *not* mirrored.

    For example:
        foo(1) < foo(0) < foo(2) is transformed to foo(2) < foo(0) < foo(1) only when strict=False

    Note that in the above statement if `foo(1) < foo(0)` is `False`, `foo(2)` will never be invoked in the
        original statement, but is always invoked in the transformed one (hence this transformation does not
        preserve semantics).

    """

    def __init__(self, p_swap: float = 1, rng: Optional[Random] = None, strict: bool = False):
        super().__init__()
        if rng is None:
            self._rng = Random()
        else:
            self._rng = rng

        assert 0 <= p_swap <= 1, "Not a valid probability."
        self._p_swap = p_swap
        self.__strict = strict

    OPPOSITES: Final = {
        cst.LessThan: cst.GreaterThan,
        cst.LessThanEqual: cst.GreaterThanEqual,
        cst.GreaterThan: cst.LessThan,
        cst.GreaterThanEqual: cst.LessThanEqual,
        cst.Equal: cst.Equal,
        cst.NotEqual: cst.NotEqual,
    }

    @staticmethod
    def get_mirrored(operator: cst.BaseCompOp) -> cst.BaseCompOp:
        return MirrorComparisons.OPPOSITES[type(operator)](
            whitespace_before=operator.whitespace_before, whitespace_after=operator.whitespace_after
        )

    def leave_Comparison(self, original_node: cst.Comparison, updated_node: cst.Comparison) -> cst.BaseExpression:
        if self._rng.random() > self._p_swap:
            return updated_node

        if any(isinstance(cmp.operator, (cst.Is, cst.IsNot, cst.In, cst.NotIn)) for cmp in updated_node.comparisons):
            return updated_node

        if self.__strict and len(updated_node.comparisons) > 1:
            return updated_node
        # We just mirror the comparisons
        comparators, operators = [updated_node.left], []
        for comparison in updated_node.comparisons:
            comparators.append(comparison.comparator)
            operators.append(comparison.operator)

        return cst.Comparison(
            left=comparators[-1],
            comparisons=[
                cst.ComparisonTarget(operator=self.get_mirrored(operator), comparator=comparator)
                for comparator, operator in zip(comparators[::-1][1:], operators[::-1])
            ],
            lpar=updated_node.lpar,
            rpar=updated_node.rpar,
        )

    @classmethod
    def apply(cls, code: str, p_apply: float = 1.0) -> str:
        module: cst.Module = cst.parse_module(code)
        wrapper = cst.MetadataWrapper(module)
        return wrapper.visit(MirrorComparisons(p_swap=p_apply)).code


class _LocalVariablesFinder(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(self):
        super().__init__()
        self.q_names = defaultdict(set)
        self.fn_cx = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self.fn_cx.append(node)
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self.fn_cx.pop()

    def visit_Name(self, node: cst.Name):
        if len(self.fn_cx) == 0:
            return
        if node.value in {"self", "cls"}:
            return
        fn_q_name: QualifiedName = tuple(self.get_metadata(QualifiedNameProvider, self.fn_cx[-1]))[0].name
        qual_names: Set[QualifiedName] = self.get_metadata(QualifiedNameProvider, node)
        for name in qual_names:
            if name.name.startswith(fn_q_name + ".<locals>") and name.source == QualifiedNameSource.LOCAL:
                self.q_names[self.fn_cx[-1]].add(name.name)


class RenameVariableInLocalScope(cst.CSTTransformer):
    """
    Rename at most one local variable in a function context, to a random string.

    This is approximately semantics-preserving within a method, but not externally, when named arguments
    are used to invoke the function.
    """

    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    CHAR_VOCAB: Final = tuple(chr(i) for i in range(97, 97 + 26)) + tuple(chr(i) for i in range(65, 65 + 26)) + ("_",)

    def __init__(
        self, qual_names: Dict[cst.FunctionDef, Set[str]], p_rename: float = 1.0, rng: Optional[Random] = None
    ):
        super().__init__()
        if rng is None:
            self._rng = Random()
        else:
            self._rng = rng

        assert 0 <= p_rename <= 1, "Not a valid probability."
        self._p_rename = p_rename
        self.__qual_names = qual_names
        self._fn_cx_stack = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if len(self.__qual_names[node]) == 0:
            self._fn_cx_stack.append(None)
            return False
        if self._rng.random() > self._p_rename:
            self._fn_cx_stack.append(None)
            return True

        to_rename = self._rng.choices(tuple(self.__qual_names[node]), k=self._rng.randint(1, 3))
        renamings = {}
        for q_name in to_rename:
            renamed_name = "".join(self._rng.choice(self.CHAR_VOCAB) for i in range(self._rng.randint(8, 15)))
            renamings[q_name] = renamed_name
        self._fn_cx_stack.append(renamings)
        return True

    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.Attribute:
        return cst.Attribute(value=original_node.value, attr=updated_node.attr)

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        if len(self._fn_cx_stack) == 0:
            return updated_node
        renamings = self._fn_cx_stack[-1]
        if renamings is None:
            return updated_node
        qual_names: Set[QualifiedName] = self.get_metadata(QualifiedNameProvider, original_node)
        if len(qual_names) == 0:
            return original_node

        node_q_name = tuple(qual_names)[0].name
        new_name = renamings.get(node_q_name)
        if new_name is not None:
            return cst.Name(new_name)
        return original_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        self._fn_cx_stack.pop()
        return updated_node

    @classmethod
    def apply(cls, code: str, p_apply: float = 1.0) -> str:
        module: cst.Module = cst.parse_module(code)
        wrapper = cst.MetadataWrapper(module)
        lvf = _LocalVariablesFinder()
        wrapper.visit(lvf)
        visitor = RenameVariableInLocalScope(lvf.q_names, p_rename=p_apply)
        return wrapper.visit(visitor).code


class CommentDelete(cst.CSTTransformer):
    def __init__(self, p_delete: float, rng: Optional[Random] = None):
        super().__init__()
        self._rng = Random() if rng is None else rng

        assert 0 <= p_delete <= 1, "Not a valid probability."
        self._p_delete = p_delete

    def leave_Comment(self, original_node: cst.Comment, updated_node: cst.Comment):
        if self._rng.random() < self._p_delete:
            return cst.RemovalSentinel.REMOVE
        else:
            return updated_node

    def leave_SimpleStatementLine(
        self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine
    ) -> Union[cst.SimpleStatementLine, cst.RemovalSentinel]:
        if (
            len(updated_node.body) == 1
            and isinstance(updated_node.body[0], cst.Expr)
            and isinstance(updated_node.body[0].value, (cst.SimpleString, cst.Comment))
            and self._rng.random() < self._p_delete
        ):
            return cst.RemovalSentinel.REMOVE
        return updated_node

    def leave_EmptyLine(
        self, original_node: cst.EmptyLine, updated_node: cst.EmptyLine
    ) -> Union[cst.EmptyLine, cst.RemovalSentinel]:
        if original_node.comment is not None and self._rng.random() > self._p_delete:
            return cst.RemovalSentinel.REMOVE
        return original_node

    @classmethod
    def apply(cls, code: str, p_apply: float = 1) -> str:
        module: cst.Module = cst.parse_module(code)
        return module.visit(CommentDelete(p_apply)).code


class NegateExpression(cst.CSTTransformer):
    def leave_BooleanOperation(
        self, original_node: cst.BooleanOperation, updated_node: cst.BooleanOperation
    ) -> cst.BooleanOperation:
        # De Morgan:
        if isinstance(original_node.operator, cst.And):
            operator = cst.Or
        elif isinstance(original_node.operator, cst.Or):
            operator = cst.And
        else:
            raise ValueError(f"Unexpected operator {original_node.operator}")

        return cst.BooleanOperation(
            lpar=original_node.lpar,
            rpar=original_node.rpar,
            left=updated_node.left,
            operator=operator(
                whitespace_before=original_node.operator.whitespace_before,
                whitespace_after=original_node.operator.whitespace_after,
            ),
            right=updated_node.right,
        )

    def leave_And(self, original_node: cst.And, updated_node: cst.And) -> cst.And:
        return updated_node

    def leave_Or(self, original_node: cst.Or, updated_node: cst.Or) -> cst.Or:
        return updated_node

    OPPOSITES: Final = {
        cst.LessThan: cst.GreaterThanEqual,
        cst.LessThanEqual: cst.GreaterThan,
        cst.GreaterThan: cst.LessThanEqual,
        cst.GreaterThanEqual: cst.LessThan,
        cst.Equal: cst.NotEqual,
        cst.NotEqual: cst.Equal,
        cst.In: cst.NotIn,
        cst.NotIn: cst.In,
        cst.Is: cst.IsNot,
        cst.IsNot: cst.Is,
    }

    def leave_Comparison(self, original_node: cst.Comparison, updated_node: cst.Comparison) -> cst.Comparison:
        return cst.Comparison(
            left=original_node.left,
            comparisons=[
                cst.ComparisonTarget(
                    operator=self.OPPOSITES[type(comparison.operator)](), comparator=comparison.comparator
                )
                for comparison in original_node.comparisons
            ],
            lpar=original_node.lpar,
            rpar=original_node.rpar,
        )

    def leave_UnaryOperation(
        self, original_node: cst.UnaryOperation, updated_node: cst.UnaryOperation
    ) -> cst.BaseExpression:
        if isinstance(original_node.operator, cst.Not):
            return original_node.expression
        else:
            return cst.UnaryOperation(cst.Not(), original_node)

    def on_visit(self, node: cst.CSTNode) -> bool:
        return isinstance(node, cst.BooleanOperation)

    def on_leave(self, original_node, updated_node):
        leave_func = getattr(self, f"leave_{type(original_node).__name__}", None)
        if (
            isinstance(original_node, (cst.BooleanOperation, cst.Comparison, cst.UnaryOperation, cst.And, cst.Or))
            and leave_func is not None
        ):
            updated_node = leave_func(original_node, updated_node)
        else:
            updated_node = cst.UnaryOperation(operator=cst.Not(), expression=original_node)
        return updated_node


class IfElseSwap(cst.CSTTransformer):
    def __init__(self, p_swap: float, rng: Optional[Random] = None):
        super().__init__()
        self._rng = Random() if rng is None else rng

        assert 0 <= p_swap <= 1, "Not a valid probability."
        self._p_swap = p_swap

    def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        if not isinstance(original_node.orelse, cst.Else):
            # No else branch
            return updated_node
        if self._rng.random() > self._p_swap:
            return updated_node

        negation_visitor = NegateExpression()
        negated = updated_node.test.visit(negation_visitor)

        return cst.If(
            test=negated,
            body=updated_node.orelse.body,
            orelse=cst.Else(
                updated_node.body,
                leading_lines=updated_node.orelse.leading_lines,
                whitespace_before_colon=updated_node.orelse.whitespace_before_colon,
            ),
            leading_lines=original_node.leading_lines,
            whitespace_before_test=original_node.whitespace_before_test,
            whitespace_after_test=original_node.whitespace_after_test,
        )

    def leave_IfExp(self, original_node: cst.IfExp, updated_node: cst.IfExp) -> cst.IfExp:
        if self._rng.random() > self._p_swap:
            return updated_node

        negation_visitor = NegateExpression()
        negated = updated_node.test.visit(negation_visitor)

        return updated_node.with_changes(test=negated, body=updated_node.orelse, orelse=updated_node.body)

    @classmethod
    def apply(cls, code: str, p_apply: float = 1.0) -> str:
        module: cst.Module = cst.parse_module(code)
        wrapper = cst.MetadataWrapper(module)
        return wrapper.visit(IfElseSwap(p_apply)).code


if __name__ == "__main__":
    import sys

    with open(sys.argv[1]) as f:
        module: cst.Module = cst.parse_module(f.read())

    wrapper = cst.MetadataWrapper(module)

    visitor = IfElseSwap(1)
    result = wrapper.visit(visitor)

    print(result.code)
