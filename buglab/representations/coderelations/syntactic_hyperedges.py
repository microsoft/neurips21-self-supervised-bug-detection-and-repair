from typing import Final, Union

import libcst as cst
from libcst.metadata import ParentNodeProvider

from buglab.representations.codereprs import PythonCodeRelations

__all__ = ["SyntacticHyperedgeRelations"]


class SyntacticHyperedgeRelations(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def __init__(self, code_relations: PythonCodeRelations):
        super().__init__()
        self.__code_relations = code_relations

    BIN_OPS: Final = {
        cst.Add: "__add__",
        cst.Subtract: "__sub__",
        cst.Multiply: "__mul__",
        cst.MatrixMultiply: "__matmul__",
        cst.Divide: "__truediv__",
        cst.FloorDivide: "__floordiv__",
        cst.Modulo: "__mod__",
        cst.Power: "__pow__",
        cst.LeftShift: "__lshift__",
        cst.RightShift: "__rshift__",
        cst.BitAnd: "__and__",
        cst.BitXor: "__xor__",
        cst.BitOr: "__or__",
    }

    BOOL_OPS: Final = {
        cst.And: "___logical_AND",
        cst.Or: "___logical_OR",
    }

    AUG_ASSIGN_OPS: Final = {
        cst.AddAssign: "__iadd__",
        cst.SubtractAssign: "__isub__",
        cst.MultiplyAssign: "__imul__",
        cst.MatrixMultiplyAssign: "__imatmul__",
        cst.DivideAssign: "__itruediv__",
        cst.FloorDivideAssign: "__ifloordiv__",
        cst.ModuloAssign: "__imod__",
        cst.PowerAssign: "__ipow__",
        cst.LeftShiftAssign: "__ilshift__",
        cst.RightShiftAssign: "__irshift__",
        cst.BitAndAssign: "__iand__",
        cst.BitXorAssign: "__ixor__",
        cst.BitOrAssign: "__ior__",
    }

    COMPARISON_OPS: Final = {
        cst.LessThan: ("__lt__", "self", "other"),
        cst.LessThanEqual: ("__lte__", "self", "other"),
        cst.Equal: ("__eq__", "self", "other"),
        cst.NotEqual: ("__ne__", "self", "other"),
        cst.GreaterThan: ("__gt__", "self", "other"),
        cst.GreaterThanEqual: ("__ge__", "self", "other"),
        cst.In: ("__contains__", "self", "item"),
        cst.NotIn: ("___not_contains__", "self", "item"),  # Not technically correct. Requires a __not__ and a negation
        cst.IsNot: ("___is_not", "item", "type"),
        cst.Is: ("___is", "item", "type"),
    }

    UNARY_OPS = {cst.Plus: "__pos__", cst.Minus: "__neg__", cst.BitInvert: "__invert__", cst.Not: "___logical_NOT"}

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> bool:
        # This assumes that the __radd__, etc are *not* invoked
        self.__code_relations.add_hyperedge_relation(
            fn_name=self.BIN_OPS[type(node.operator)],
            fn_docstring=None,
            arguments={"self": node.left, "other": node.right},
            return_value=node,
        )
        return True

    def visit_AugAssign(self, node: cst.AugAssign) -> bool:
        self.__code_relations.add_hyperedge_relation(
            fn_name=self.AUG_ASSIGN_OPS[type(node.operator)],
            fn_docstring=None,
            arguments={"self": node.target, "other": node.value},
            return_value=node,
        )
        return True

    def visit_BooleanOperation(self, node: cst.BooleanOperation) -> bool:
        self.__code_relations.add_hyperedge_relation(
            fn_name=self.BOOL_OPS[type(node.operator)],
            fn_docstring=None,
            arguments={"left": node.left, "right": node.right},
            return_value=node,
        )
        return True

    def visit_UnaryOperation(self, node: cst.UnaryOperation) -> bool:
        self.__code_relations.add_hyperedge_relation(
            fn_name=self.UNARY_OPS[type(node.operator)],
            fn_docstring=None,
            arguments={"self": node.expression},
            return_value=node,
        )
        return True

    def visit_Comparison(self, node: cst.Comparison) -> bool:
        left_node = node.left
        for comparison in node.comparisons:
            fn_name, arg1_name, arg2_name = self.COMPARISON_OPS[type(comparison.operator)]
            self.__code_relations.add_hyperedge_relation(
                fn_name=fn_name,
                fn_docstring=None,
                arguments={
                    arg1_name: left_node,
                    arg2_name: comparison.comparator,
                },
                return_value=comparison,
            )
            left_node = comparison.comparator

        if len(node.comparisons) > 1:
            # Combine comparison targets, with a set-and
            self.__code_relations.add_hyperedge_relation(
                fn_name=self.BOOL_OPS[cst.And],
                fn_docstring=None,
                arguments={"args": set(node.comparisons)},
                return_value=node,
            )
        else:
            self.__code_relations.add_hyperedge_relation(
                fn_name="___assign",  # (not a real function)
                fn_docstring=None,
                arguments={"source": node.comparisons[0]},
                return_value=node,
            )
        return True

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        parent_node = self.get_metadata(ParentNodeProvider, node)
        if not isinstance(parent_node, cst.AssignTarget):
            # Assignments are handled at the level of the assignment
            self.__code_relations.add_hyperedge_relation(
                fn_name="__getattribute__",
                fn_docstring=None,
                arguments={"self": node.value, "name": node.attr},
                return_value=node,
            )
        return True

    def visit_Subscript(self, node: cst.Subscript) -> bool:
        parent_node = self.get_metadata(ParentNodeProvider, node)
        if not isinstance(parent_node, cst.AssignTarget):
            # Assignments are handled at the level of the assignment
            self.__code_relations.add_hyperedge_relation(
                fn_name="__getitem__",
                fn_docstring=None,
                arguments={"self": node.value, "key": node.slice},
                return_value=node,
            )
        return True

    def visit_Assign(self, node: cst.Assign) -> bool:
        # Multiple targets are of the form target1 = target2 = value
        for assign_target in node.targets:
            self._assign_op(node, assign_target)
        return True

    def visit_NamedExpr(self, node: cst.NamedExpr) -> bool:
        self._assign_op(node, node)
        return True

    def _assign_op(self, node: Union[cst.Assign, cst.NamedExpr], assign_target: Union[cst.AssignTarget, cst.NamedExpr]):
        if isinstance(assign_target.target, cst.Attribute):
            self.__code_relations.add_hyperedge_relation(
                fn_name="__setattr__",
                fn_docstring=None,
                arguments={"self": assign_target.target.value, "name": assign_target.target.attr, "value": node.value},
                return_value=None,
            )
        elif (
            isinstance(node.value, (cst.Tuple, cst.List))
            and isinstance(assign_target.target, (cst.Tuple, cst.List))
            and len(assign_target.target.elements) == len(node.value.elements)
        ):
            # Explicitly deconstruct assignment
            for target_element, value_element in zip(assign_target.target.elements, node.value.elements):
                self.__code_relations.add_hyperedge_relation(
                    fn_name="___assign",  # (not a real function)
                    fn_docstring=None,
                    arguments={"source": value_element.value},
                    return_value=target_element.value,
                )
        elif isinstance(assign_target.target, (cst.Name, cst.List, cst.Tuple)):
            self.__code_relations.add_hyperedge_relation(
                fn_name="___assign",  # (not a real function)
                fn_docstring=None,
                arguments={"source": node.value},
                return_value=assign_target.target,
            )
        elif isinstance(assign_target.target, cst.Subscript):
            self.__code_relations.add_hyperedge_relation(
                fn_name="__setitem__",
                fn_docstring=None,
                arguments={
                    "self": assign_target.target.value,
                    "key": set(assign_target.target.slice),  # set is not exactly right...
                    "value": node.value,
                },
                return_value=None,
            )
        else:
            raise Exception("Unrecognized type of AssignTarget.")

    @staticmethod
    def add_relations(code_rels: PythonCodeRelations, *args) -> None:
        code_rels.ast_with_metadata_wrapper.visit(SyntacticHyperedgeRelations(code_rels))
