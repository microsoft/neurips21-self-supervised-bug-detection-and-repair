from itertools import chain
from typing import List, Set, Tuple

import libcst as cst

from buglab.representations.codereprs import PythonCodeRelations

__all__ = ["ControlFlow"]


class ControlFlow(cst.CSTVisitor):
    def __init__(self, code_relations: PythonCodeRelations):
        super().__init__()
        self.__code_relations = code_relations

        # Control flow
        self.__last_statements: Tuple[...] = tuple()
        self.__returns_from: List[cst.CSTNode] = []
        self.__yields_from: List[cst.CSTNode] = []
        self.__continue_from: Set[cst.CSTNode] = set()
        self.__break_from: Set[cst.CSTNode] = set()

    def _add_next(self, to_node: cst.CSTNode) -> None:
        for stmt in self.__last_statements:
            self.__code_relations.add_relation("ControlFlowNext", stmt, to_node)
        self.__last_statements = (to_node,)

    def _add_assigned_from(self, expression: cst.CSTNode, target: cst.CSTNode):
        self.__code_relations.add_relation("AssignedFrom", target, expression)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        outside_last_statements, self.__last_statements = self.__last_statements, (node,)
        outside_returns_cx, self.__returns_from = self.__returns_from, []
        outside_yields_cx, self.__yields_from = self.__yields_from, []

        node.body.visit(self)

        for return_stmt in chain(self.__returns_from, self.__last_statements):
            self.__code_relations.add_relation("ReturnsFrom", node, return_stmt)
        for yield_stmt in self.__yields_from:
            self.__code_relations.add_relation("YieldsFrom", node, yield_stmt)

        self.__last_statements = outside_last_statements
        self.__returns_from = outside_returns_cx
        self.__yields_from = outside_yields_cx
        return False

    def visit_Return(self, node: cst.Return) -> bool:
        if node.value is not None:
            node.value.visit(self)
            self._add_next(node)
        self.__returns_from.append(node)
        self.__last_statements = tuple()  # Stop control flow
        return False

    def visit_Yield(self, node: cst.Yield) -> bool:
        if node.value is not None:
            node.value.visit(self)
            self._add_next(node)
            self.__yields_from.append(node)
        return False

    def visit_Import(self, node: cst.Import) -> bool:
        self._add_next(node)
        return False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        self._add_next(node)
        return False

    def visit_Call(self, node: cst.Call) -> bool:
        node.func.visit(self)
        for arg in node.args:
            arg.visit(self)
        self._add_next(node)
        return False

    def visit_Assign(self, node: cst.Assign) -> bool:
        node.value.visit(self)
        for target in node.targets:
            self._add_assigned_from(node.value, target)
        self._add_next(node)
        return False

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        if node.value is not None:
            node.value.visit(self)
            self._add_assigned_from(node.value, node.target)
        self._add_next(node)
        return False

    def visit_NamedExpr(self, node: cst.NamedExpr) -> bool:
        node.value.visit(self)
        self._add_assigned_from(node.value, node.target)
        self._add_next(node)
        return False

    def visit_AugAssign(self, node: cst.AugAssign) -> bool:
        node.value.visit(self)
        self._add_assigned_from(node.value, node.target)
        self._add_next(node)
        return False

    def visit_If(self, node: cst.If) -> bool:
        node.test.visit(self)
        after_test = self.__last_statements

        node.body.visit(self)
        then_out, self.__last_statements = self.__last_statements, after_test

        if node.orelse is not None:
            node.orelse.visit(self)

        self.__last_statements = self.__last_statements + then_out
        return False

    def visit_IfExp(self, node: cst.IfExp) -> bool:
        node.test.visit(self)
        after_test = self.__last_statements

        node.body.visit(self)
        then_out, self.__last_statements = self.__last_statements, after_test

        node.orelse.visit(self)

        self.__last_statements = self.__last_statements + then_out
        return False

    def visit_Try(self, node: cst.Try) -> bool:
        starting_statements = self.__last_statements
        node.body.visit(self)
        if node.orelse is not None:
            node.orelse.visit(self)

        # The common heuristic. Treat exceptions as if-like statements.
        exception_starting_statements = self.__last_statements + starting_statements
        out_last_statements = tuple()
        for exc_handler in node.handlers:
            exc_handler.visit(self)
            out_last_statements += self.__last_statements
            self.__last_statements = exception_starting_statements

        self.__last_statements += out_last_statements

        if node.finalbody is not None:
            node.finalbody.visit(self)
        return False

    def visit_Continue(self, node: cst.Continue) -> bool:
        self._add_next(node)
        self.__continue_from.add(node)
        self.__last_statements = tuple()
        return False

    def visit_Break(self, node: cst.Break) -> bool:
        self._add_next(node)
        self.__break_from.add(node)
        self.__last_statements = tuple()
        return False

    def visit_While(self, node: cst.While) -> bool:
        prev_continue, prev_break = self.__continue_from, self.__break_from
        self.__continue_from, self.__break_from = set(), set()

        node.test.visit(self)

        node.body.visit(self)
        self.__last_statements += tuple(self.__continue_from)
        node.test.visit(self)

        if node.orelse is not None:
            node.orelse.visit(self)

        self.__last_statements += tuple(self.__break_from)

        self.__continue_from, self.__break_from = prev_continue, prev_break
        return False

    def visit_For(self, node: cst.For) -> bool:
        prev_continue, prev_break = self.__continue_from, self.__break_from
        self.__continue_from, self.__break_from = set(), set()

        node.iter.visit(self)
        self._add_assigned_from(node.iter, node.target)
        node.body.visit(self)
        self.__last_statements += tuple(self.__continue_from)
        node.body.visit(self)

        if node.orelse is not None:
            node.orelse.visit(self)

        self.__last_statements += tuple(self.__break_from)

        self.__continue_from, self.__break_from = prev_continue, prev_break
        return False

    def visit_IndentedBlock(self, node: cst.IndentedBlock) -> bool:
        for stmt in node.body:
            stmt.visit(self)
        return False

    def visit_WithItem(self, node: cst.WithItem) -> bool:
        node.item.visit(self)
        if node.asname is not None:
            self._add_assigned_from(node.item, node.asname.name)
            node.asname.visit(self)
        return False

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> bool:
        node.left.visit(self)
        node.right.visit(self)
        self._add_next(node)
        return False

    def visit_BooleanOperation(self, node: cst.BooleanOperation) -> bool:
        node.left.visit(self)
        node.right.visit(self)
        self._add_next(node)
        return False

    def visit_Comparison(self, node: cst.Comparison) -> bool:
        node.left.visit(self)
        for cmp_target in node.comparisons:
            self._add_next(cmp_target)
            cmp_target.visit(self)  # TODO: Visit as optional
        return False

    def visit_Assert(self, node: cst.Assert) -> bool:
        node.test.visit(self)
        self._add_next(node)
        return False

    def visit_UnaryOperation(self, node: cst.UnaryOperation) -> bool:
        node.expression.visit(self)
        self._add_next(node)
        return False

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        node.value.visit(self)
        node.attr.visit(self)
        return False

    def visit_ListComp(self, node: cst.ListComp) -> bool:
        node.for_in.visit(self)
        node.elt.visit(self)
        return False

    def visit_SetComp(self, node: cst.SetComp) -> bool:
        node.for_in.visit(self)
        node.elt.visit(self)
        return False

    def visit_DictComp(self, node: cst.DictComp) -> bool:
        node.for_in.visit(self)
        node.key.visit(self)
        node.value.visit(self)
        return False

    def visit_GeneratorExp(self, node: cst.GeneratorExp) -> bool:
        node.for_in.visit(self)
        node.elt.visit(self)
        return False

    @staticmethod
    def compute_controlflow_relations(code_rels: PythonCodeRelations, *args) -> None:
        code_rels.ast_with_metadata_wrapper.visit(ControlFlow(code_rels))
