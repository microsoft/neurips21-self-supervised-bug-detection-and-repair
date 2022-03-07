from typing import DefaultDict, FrozenSet, Set

import libcst as cst
from collections import defaultdict
from copy import copy
from itertools import chain
from libcst.metadata import QualifiedName, QualifiedNameProvider, QualifiedNameSource

from buglab.representations.codereprs import PythonCodeRelations


class DataFlow(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(self, code_relations: PythonCodeRelations):
        super().__init__()
        self.__code_relations = code_relations

        self.__last_access: DefaultDict[QualifiedName, FrozenSet[cst.CSTNode]] = defaultdict(frozenset)
        self.__last_write: DefaultDict[QualifiedName, FrozenSet[cst.CSTNode]] = defaultdict(frozenset)

        self.__returns_from_access = []
        self.__continue_from_access = []
        self.__break_from_access = []

        self.__in_write_context = False

    def visit_Name(self, node: cst.Name):
        if self.__in_write_context:
            self.__record_write(node)
        else:
            self.__record_access(node)

    def visit_Attribute(self, node: cst.Attribute):
        if self.__in_write_context:
            self.__record_write(node)
        else:
            self.__record_access(node)

    def __record_access(self, node: cst.CSTNode) -> None:
        """Add a link to the symbol node and record access."""
        qual_names = self.get_metadata(QualifiedNameProvider, node)
        for qname in qual_names:
            self.__code_relations.add_relation("OccurrenceOf", node, metadata=qname.name)

            if qname.source != QualifiedNameSource.LOCAL:
                continue
            for last_accessed_node in self.__last_access[qname]:
                self.__code_relations.add_relation("NextMayUse", last_accessed_node, node)
            for last_write_node in self.__last_write[qname]:
                self.__code_relations.add_relation("LastMayWrite", last_write_node, node)
            self.__last_access[qname] = frozenset({node})

    def __record_write(self, node: cst.CSTNode) -> None:
        qual_names = self.get_metadata(QualifiedNameProvider, node)
        for qname in qual_names:
            self.__code_relations.add_relation("OccurrenceOf", node, metadata=qname.name)

            self.__last_access[qname] = frozenset()
            self.__last_write[qname] = frozenset({node})

    def visit_in_write_context(self, node: cst.CSTNode) -> None:
        try:
            self.__in_write_context = True
            node.visit(self)
        finally:
            self.__in_write_context = False

    def _add_dummy_end_nodes(self) -> None:
        for qname, last_accesses in self.__last_access.items():
            for access in last_accesses:
                self.__code_relations.add_relation("MayFinalUseOf", access, metadata=qname.name)

    def __merge_last_access(
        self, accesses1: DefaultDict[str, FrozenSet[cst.CSTNode]], accesses2: DefaultDict[str, FrozenSet[cst.CSTNode]]
    ) -> DefaultDict[str, FrozenSet[cst.CSTNode]]:
        all_vars = set(accesses1) | set(accesses2)
        return defaultdict(frozenset, {v: frozenset(chain(accesses1[v], accesses2[v])) for v in all_vars})

    def visit_If(self, node: cst.If) -> bool:
        node.test.visit(self)
        last_access_snapshot = copy(self.__last_access)
        node.body.visit(self)
        self.__last_access, after_body_snapshot = last_access_snapshot, self.__last_access
        if node.orelse is not None:
            node.orelse.visit(self)
        self.__last_access = self.__merge_last_access(self.__last_access, after_body_snapshot)
        return False

    def visit_IfExp(self, node: cst.IfExp) -> bool:
        node.test.visit(self)

        last_access_snapshot = copy(self.__last_access)
        node.body.visit(self)

        self.__last_access, after_body_snapshot = last_access_snapshot, self.__last_access
        node.orelse.visit(self)
        self.__last_access = self.__merge_last_access(self.__last_access, after_body_snapshot)
        return False

    def visit_Return(self, node: cst.Return) -> bool:
        if node.value is not None:
            node.value.visit(self)
        self.__returns_from_access[-1] = self.__merge_last_access(self.__returns_from_access[-1], self.__last_access)
        self.__last_access = defaultdict(frozenset)
        return False

    def visit_Continue(self, node: cst.Continue) -> bool:
        self.__continue_from_access[-1] = self.__merge_last_access(self.__continue_from_access[-1], self.__last_access)
        self.__last_access = defaultdict(frozenset)
        return False

    def visit_Break(self, node: cst.Break) -> bool:
        self.__break_from_access[-1] = self.__merge_last_access(self.__break_from_access[-1], self.__last_access)
        self.__last_access = defaultdict(frozenset)
        return False

    def visit_While(self, node: cst.While) -> bool:
        node.test.visit(self)

        accesses_after_first_test = copy(self.__last_access)

        self.__break_from_access.append(defaultdict(frozenset))
        self.__continue_from_access.append(defaultdict(frozenset))

        node.body.visit(self)
        self.__last_access = self.__merge_last_access(self.__continue_from_access.pop(), self.__last_access)

        self.__break_from_access[-1] = defaultdict(frozenset)  # Reset break
        self.__continue_from_access.append(defaultdict(frozenset))

        node.test.visit(self)
        node.body.visit(self)
        self.__last_access = self.__merge_last_access(self.__continue_from_access.pop(), self.__last_access)
        node.test.visit(self)

        self.__last_access = self.__merge_last_access(self.__last_access, accesses_after_first_test)
        if node.orelse is not None:
            node.orelse.visit(self)

        self.__last_access = self.__merge_last_access(self.__last_access, self.__break_from_access.pop())
        return False

    def visit_For(self, node: cst.For) -> bool:
        node.iter.visit(self)

        accesses_after_zero_iterations = copy(self.__last_access)

        self.__break_from_access.append(defaultdict(frozenset))
        self.__continue_from_access.append(defaultdict(frozenset))

        self.visit_in_write_context(node.target)

        node.body.visit(self)
        self.__last_access = self.__merge_last_access(self.__continue_from_access.pop(), self.__last_access)

        self.__break_from_access[-1] = defaultdict(frozenset)  # Reset break
        self.__continue_from_access.append(defaultdict(frozenset))

        self.visit_in_write_context(node.target)
        node.body.visit(self)

        self.__last_access = self.__merge_last_access(self.__continue_from_access.pop(), self.__last_access)

        self.__last_access = self.__merge_last_access(self.__last_access, accesses_after_zero_iterations)
        if node.orelse is not None:
            node.orelse.visit(self)

        self.__last_access = self.__merge_last_access(self.__last_access, self.__break_from_access.pop())
        return False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        qual_names = self.get_metadata(QualifiedNameProvider, node.name)
        for qname in qual_names:
            self.__code_relations.add_relation("OccurrenceOf", node, metadata=qname.name)

        self.__returns_from_access.append(defaultdict(frozenset))
        self.visit_in_write_context(node.params)
        node.body.visit(self)
        self.__last_access = self.__merge_last_access(self.__last_access, self.__returns_from_access.pop())
        return False

    def visit_Assign(self, node: cst.Assign) -> bool:
        for target in node.targets:
            self.visit_in_write_context(target)
        node.value.visit(self)
        return False

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        self.visit_in_write_context(node.target)
        if node.value is not None:
            node.value.visit(self)
        return False

    def visit_AugAssign(self, node: cst.AugAssign) -> bool:
        self.visit_in_write_context(node.target)
        node.value.visit(self)
        return False

    def visit_NamedExpr(self, node: cst.NamedExpr) -> bool:
        self.visit_in_write_context(node.target)
        node.value.visit(self)
        return False

    def visit_Subscript(self, node: cst.Subscript) -> bool:
        if self.__in_write_context:
            node.value.visit(self)
            self.__in_write_context = False
            for s in node.slice:
                s.visit(self)
            self.__in_write_context = True
            return False
        return True

    def visit_AsName(self, node: cst.AsName) -> bool:
        self.visit_in_write_context(node.name)
        return False

    def visit_Lambda(self, node: cst.Lambda) -> bool:
        self.visit_in_write_context(node.params)
        node.body.visit(self)
        return False

    def visit_CompFor(self, node: cst.CompFor) -> bool:
        node.iter.visit(self)
        self.visit_in_write_context(node.target)
        for if_cnd in node.ifs:
            if_cnd.visit(self)
        if node.inner_for_in is not None:
            node.inner_for_in.visit(self)
        return False

    @staticmethod
    def compute_dataflow_relations(code_rels: PythonCodeRelations, *args) -> None:
        dflow = DataFlow(code_rels)
        code_rels.ast_with_metadata_wrapper.visit(dflow)
        dflow._add_dummy_end_nodes()
