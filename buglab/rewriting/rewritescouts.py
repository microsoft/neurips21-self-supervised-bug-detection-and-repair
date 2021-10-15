from abc import ABC
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Tuple, Type, Union

import libcst as cst
from libcst.metadata import CodePosition, CodeRange, ParentNodeProvider, PositionProvider, Scope, ScopeProvider
from libcst.metadata.scope_provider import (
    BuiltinAssignment,
    ComprehensionScope,
    FunctionScope,
    GlobalScope,
    QualifiedNameSource,
)
from typing_extensions import Final

from .rewriteops import AbstractRewriteOp, ArgSwapOp, ReplaceCodeTextOp


class ICodeRewriteScout(cst.BatchableCSTVisitor, ABC):
    """
    Interface of a class that finds the locations where a code rewrite can be applied.
    All possible rewrites are accumulated in the `op_store` list passed in the constructor.
    The `op_metadata_store` optionally stores additional metadata.
    """

    def __init__(
        self,
        op_store: List[AbstractRewriteOp],
        op_metadata_store: List[Tuple[Type["ICodeRewriteScout"], cst.CSTNode, Any]] = None,
    ):
        super().__init__()
        self.__ops = op_store
        self.__op_metadata_store = op_metadata_store

    def add_mod_op(self, op: AbstractRewriteOp, metadata: Tuple[Type["ICodeRewriteScout"], cst.CSTNode, Any]) -> None:
        self.__ops.append(op)
        if self.__op_metadata_store is not None:
            self.__op_metadata_store.append(metadata)


class BinaryOperatorRewriteScout(ICodeRewriteScout):
    """Introduce a bug by replacing a binary operator."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    BinOps: Final = frozenset(
        {
            cst.Add,
            cst.Subtract,
            cst.Multiply,
            cst.Power,
            cst.Divide,
            cst.FloorDivide,
            cst.Modulo,
            cst.MatrixMultiply,
            cst.LeftShift,
            cst.RightShift,
            cst.BitOr,
            cst.BitAnd,
            cst.BitXor,
        }
    )

    DISABLED_BINOPS = frozenset()

    def visit_BinaryOperation(self, node: cst.BinaryOperation):
        assert type(node.operator) in self.BinOps
        if type(node.operator) in self.DISABLED_BINOPS:
            return

        op_range: CodeRange = self.get_metadata(PositionProvider, node.operator)
        for op in self.BinOps:
            if isinstance(node.operator, op) or op in self.DISABLED_BINOPS:
                continue
            replace_op = ReplaceCodeTextOp(op_range, op._get_token(op))
            self.add_mod_op(replace_op, (BinaryOperatorRewriteScout, node, None))


class AssignRewriteScout(ICodeRewriteScout):
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider)

    AugAssignOps: Final = frozenset(
        {
            cst.AddAssign,
            cst.SubtractAssign,
            cst.MultiplyAssign,
            cst.PowerAssign,
            cst.DivideAssign,
            cst.FloorDivideAssign,
            cst.ModuloAssign,
            cst.MatrixMultiplyAssign,
            cst.LeftShiftAssign,
            cst.RightShiftAssign,
            cst.BitOrAssign,
            cst.BitAndAssign,
            cst.BitXorAssign,
        }
    )

    DISABLED_OPS: FrozenSet[cst.BaseAugOp] = frozenset()

    def visit_AugAssign(self, node: cst.AugAssign):
        assert type(node.operator) in self.AugAssignOps
        if type(node.operator) in self.DISABLED_OPS:
            return

        op_range: CodeRange = self.get_metadata(PositionProvider, node.operator)
        for op in self.AugAssignOps:
            if isinstance(node.operator, op) or op in self.DISABLED_OPS:
                continue
            replace_op = ReplaceCodeTextOp(op_range, op._get_token(op))
            self.add_mod_op(replace_op, (AssignRewriteScout, node, None))

        # Also replace with a standard assignment
        replace_op = ReplaceCodeTextOp(op_range, "=")
        self.add_mod_op(replace_op, (AssignRewriteScout, node, None))

    def has_assignments_before_node(
        self, name: cst.Name, this_assignment_node: Union[cst.AnnAssign, cst.Assign]
    ) -> bool:
        scope_data = self.get_metadata(ScopeProvider, name)
        this_assignment_location: CodeRange = self.get_metadata(PositionProvider, this_assignment_node)

        for assignment in scope_data.assignments[name]:
            if isinstance(assignment, BuiltinAssignment):
                return True
            assignment_range: CodeRange = self.get_metadata(PositionProvider, assignment.node)
            # Very rough heuristic for use-before-def
            if assignment_range.start.line < this_assignment_location.start.line:
                return True
        return False

    def visit_Assign(self, node: cst.Assign):
        if len(node.targets) != 1:
            return

        if isinstance(node.targets[0].target, cst.Name) and not self.has_assignments_before_node(
            node.targets[0].target, node
        ):
            return

        last_target_location: CodeRange = self.get_metadata(PositionProvider, node.targets[-1])
        op_range = CodeRange(
            (
                last_target_location.end.line,
                last_target_location.end.column + len(node.targets[-1].whitespace_before_equal.value),
            ),
            (
                last_target_location.end.line,
                last_target_location.end.column + len(node.targets[-1].whitespace_before_equal.value) + 1,
            ),
        )

        for op in self.AugAssignOps:
            if op in self.DISABLED_OPS:
                continue
            replace_op = ReplaceCodeTextOp(op_range, op._get_token(op))
            self.add_mod_op(replace_op, (AssignRewriteScout, node, None))


class BooleanOperatorRewriteScout(ICodeRewriteScout):
    boolean_op_replacements: Final = {cst.And: frozenset({cst.Or}), cst.Or: frozenset({cst.And})}

    METADATA_DEPENDENCIES = (PositionProvider,)

    def visit_BooleanOperation(self, node: cst.BooleanOperation):
        assert type(node.operator) in self.boolean_op_replacements
        op_range: CodeRange = self.get_metadata(PositionProvider, node.operator)
        for op in self.boolean_op_replacements[type(node.operator)]:
            replace_op = ReplaceCodeTextOp(op_range, op._get_token(op))
            self.add_mod_op(replace_op, (BooleanOperatorRewriteScout, node, None))

        self.__rewrite_test(node.left)
        self.__rewrite_test(node.right)

    # Only negate nodes where pre-pending a 'not' (textually) does not need a parenthesis
    NEGATABLE_NODE_TYPES: Final = (cst.Name, cst.Call)

    def __rewrite_test(self, test_node: cst.BaseExpression):
        if (
            isinstance(test_node, cst.UnaryOperation)
            and isinstance(test_node.operator, cst.Not)
            and isinstance(test_node.expression, self.NEGATABLE_NODE_TYPES)
        ):
            operator_range = self.get_metadata(PositionProvider, test_node.operator)
            replace_op = ReplaceCodeTextOp(
                CodeRange(
                    operator_range.start, CodePosition(operator_range.start.line, operator_range.start.column + 4)
                ),
                "",
            )
        elif isinstance(test_node, self.NEGATABLE_NODE_TYPES):
            test_range: CodeRange = self.get_metadata(PositionProvider, test_node)
            replace_op = ReplaceCodeTextOp(CodeRange(test_range.start, test_range.start), "not ")
        else:
            return
        self.add_mod_op(replace_op, (BooleanOperatorRewriteScout, test_node, None))

    def visit_If(self, node: cst.If):
        self.__rewrite_test(node.test)

    def visit_While(self, node: cst.While):
        self.__rewrite_test(node.test)


# Out-of-place patch of libcst until this is fixed in the upstream package
# https://github.com/Instagram/LibCST/issues/315
def codegen_impl_fixed(self, state) -> None:
    self.whitespace_before._codegen(state)
    with state.record_syntactic_position(self):
        state.add_token(self.value)
    self.whitespace_after._codegen(state)


cst.NotEqual._codegen_impl = codegen_impl_fixed


class ComparisonOperatorRewriteScout(ICodeRewriteScout):
    ALL_OPERATORS: Final = {
        cst.LessThan: ("<", False),
        cst.LessThanEqual: ("<=", False),
        cst.GreaterThan: (">", False),
        cst.GreaterThanEqual: (">=", False),
        cst.Equal: ("==", False),
        cst.NotEqual: ("!=", False),
        cst.In: ("in", True),
        cst.NotIn: ("not in", True),
        cst.Is: ("is", True),
        cst.IsNot: ("is not", True),
    }

    OPERATOR_SWAP_GROUPS: Final = frozenset(
        [
            frozenset(
                {cst.LessThan, cst.LessThanEqual, cst.GreaterThan, cst.GreaterThanEqual, cst.Equal, cst.NotEqual}
            ),
            frozenset({cst.In, cst.NotIn}),
            frozenset({cst.Is, cst.IsNot}),
        ]
    )

    METADATA_DEPENDENCIES = (PositionProvider,)

    def visit_ComparisonTarget(self, node: cst.ComparisonTarget):
        operator_node = node.operator
        current_op_type = type(operator_node)
        _, original_needs_space = self.ALL_OPERATORS[current_op_type]

        assert current_op_type in self.ALL_OPERATORS
        op_range: CodeRange = self.get_metadata(PositionProvider, operator_node)

        start_pos, end_pos = op_range.start, op_range.end
        if original_needs_space:
            start_pos = CodePosition(start_pos.line, start_pos.column - 1)
            end_pos = CodePosition(end_pos.line, end_pos.column + 1)

        # Find swap group
        for group in self.OPERATOR_SWAP_GROUPS:
            if current_op_type in group:
                target_swap_group = group
                break
        else:
            raise Exception("Operator not found.")

        for op in target_swap_group:
            if op is current_op_type:
                continue

            operator_token, replacement_requires_space = self.ALL_OPERATORS[op]

            if replacement_requires_space:
                space_before, space_after = " ", " "
            else:
                space_before, space_after = "", ""

            replace_op = ReplaceCodeTextOp(CodeRange(start_pos, end_pos), space_before + operator_token + space_after)
            self.add_mod_op(replace_op, (ComparisonOperatorRewriteScout, node, None))


class LiteralRewriteScout(ICodeRewriteScout):
    METADATA_DEPENDENCIES = (
        PositionProvider,
        ParentNodeProvider,
    )

    def visit_UnaryOperation(self, node: cst.UnaryOperation):
        try:
            if isinstance(node.operator, cst.Minus) and isinstance(node.expression, cst.Integer):
                int_val = -int(node.expression.value)
                self.__replace_int_literal(int_val, node)
            elif isinstance(node.operator, cst.Plus) and isinstance(node.expression, cst.Integer):
                pass  # This is a non-reversible edit (we cannot know when we see "-1" if it came for a "1" or a "+1")
        except ValueError:
            pass

    def visit_Integer(self, node: cst.Integer):
        parent = self.get_metadata(ParentNodeProvider, node)
        if isinstance(parent, cst.UnaryOperation):
            return
        if isinstance(parent, cst.Param):
            return
        try:
            int_val = int(node.value)
            self.__replace_int_literal(int_val, node)
        except ValueError:
            pass

    def __replace_int_literal(self, int_val, node):
        op_range: CodeRange = self.get_metadata(PositionProvider, node)
        target_values = {-2, -1, 0, 1, 2}
        if int_val not in target_values:
            return  # Not reversible otherwise
        for target_value in target_values:
            if target_value == int_val:
                continue
            replace_op = ReplaceCodeTextOp(op_range, str(target_value))
            self.add_mod_op(replace_op, (LiteralRewriteScout, node, None))

    def visit_Name(self, node: cst.Name):
        op_range: CodeRange = self.get_metadata(PositionProvider, node)

        parent = self.get_metadata(ParentNodeProvider, node)
        if isinstance(parent, cst.Param):
            return

        if node.value == "True":
            replace_op = ReplaceCodeTextOp(op_range, "False")
            self.add_mod_op(replace_op, (LiteralRewriteScout, node, None))
        elif node.value == "False":
            replace_op = ReplaceCodeTextOp(op_range, "True")
            self.add_mod_op(replace_op, (LiteralRewriteScout, node, None))


class VariableMisuseRewriteScout(ICodeRewriteScout):
    METADATA_DEPENDENCIES = (PositionProvider, ParentNodeProvider, ScopeProvider)

    def __init__(
        self, op_store: List[AbstractRewriteOp], op_metadata_store: List[Tuple[Type["ICodeRewriteScout"], Any]] = None
    ):
        super().__init__(op_store, op_metadata_store)
        self.__enabled = True

    BLACKLISTED_NAMES: Final = frozenset({"self"})

    def __is_local_symbol(self, scope: Scope, name: str) -> bool:
        return any(qn.source == QualifiedNameSource.LOCAL for qn in scope.get_qualified_names_for(name))

    def __all_possible_var_misuses(self, scope: Scope) -> Dict[str, str]:
        all_names = {}
        for name in scope._assignments:
            if name in self.BLACKLISTED_NAMES or name in all_names:
                continue
            for qn in scope.get_qualified_names_for(name):
                if qn.source == QualifiedNameSource.LOCAL:
                    all_names[name] = qn.name

        if isinstance(scope, ComprehensionScope):
            # Get the closure of the comprehension
            all_names.update(self.__all_possible_var_misuses(scope.parent))
        return all_names

    def get_parent_statement(self, node: cst.CSTNode) -> cst.CSTNode:
        while not isinstance(node, cst.BaseStatement):
            node = self.get_metadata(ParentNodeProvider, node)
        return node

    def visit_Name(self, node: cst.Name):
        if not self.__enabled or node.value in self.BLACKLISTED_NAMES:
            return

        try:
            scope_data: Scope = self.get_metadata(ScopeProvider, node)
        except:
            return

        if not self.__is_local_symbol(scope_data, node.value):
            return

        is_access = node in scope_data.accesses
        if not is_access and len(scope_data.accesses[node]) > 0:
            return

        statement_of_this_node = self.get_parent_statement(node)
        statement_location = self.get_metadata(PositionProvider, statement_of_this_node)

        if not isinstance(scope_data, FunctionScope) and not isinstance(scope_data, ComprehensionScope):
            # Look only for names used within functions or comprehensions
            return

        scope_data: Union[FunctionScope, ComprehensionScope]
        accesses = scope_data[node.value]

        if any(isinstance(a.scope, GlobalScope) for a in accesses):
            return
        candidate_misuses = self.__all_possible_var_misuses(scope_data)

        # Filter use-before-def
        def has_assignments_before_node(name: str):
            for assignment in scope_data.assignments[name]:
                assignment_node: cst.CSTNode = self.get_parent_statement(assignment.node)
                assignment_location: CodeRange = self.get_metadata(PositionProvider, assignment_node)
                # A mostly gross over-simplified heuristic to avoid use-before-def
                if assignment_location.start.line < statement_location.start.line:
                    return True
            return False

        name_range: CodeRange = self.get_metadata(PositionProvider, node)

        if not has_assignments_before_node(node.value):
            return
        if node.value not in candidate_misuses:
            return  # TODO: This shouldn't happen. Debug
        for name in candidate_misuses:
            if name == node.value:
                continue
            if not has_assignments_before_node(name):
                continue
            self.add_mod_op(
                ReplaceCodeTextOp(name_range, name), (VariableMisuseRewriteScout, node, candidate_misuses[name])
            )

    def visit_Parameters(self, node: cst.Parameters):
        self.__enabled = False

    def leave_Parameters(self, original_node: cst.Parameters) -> None:
        self.__enabled = True


class ArgSwapRewriteScout(ICodeRewriteScout):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def visit_Call(self, node: cst.Call):
        non_param_args = [i for i, arg in enumerate(node.args) if arg.keyword is None and arg.star == ""]
        if len(non_param_args) <= 1:
            return

        func_range: CodeRange = self.get_metadata(PositionProvider, node.func)
        for i, j in combinations(non_param_args, 2):
            self.add_mod_op(ArgSwapOp(func_range, (i, j)), (ArgSwapRewriteScout, node, None))
