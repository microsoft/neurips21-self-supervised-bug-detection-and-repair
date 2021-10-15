from collections.abc import Sequence
from dataclasses import fields

import libcst as cst

from buglab.representations.codereprs import PythonCodeRelations
from buglab.utils.cstutils import is_whitespace_node

__all__ = ["AstRelations"]


class AstRelations(cst.CSTVisitor):
    def __init__(self, code_relations: PythonCodeRelations):
        super().__init__()
        self.__code_relations = code_relations

    def __named_children(self, node: cst.CSTNode):
        names = {}
        for f in fields(node):
            child_attribute = getattr(node, f.name)
            if isinstance(child_attribute, Sequence):
                for c in child_attribute:
                    names[c] = f.name
            else:
                names[child_attribute] = f.name

        for child in node.children:
            yield child, names[child]

    def on_visit(self, node: cst.CSTNode) -> bool:
        if is_whitespace_node(node):
            return False
        prev_child = None

        for child, child_name in self.__named_children(node):
            if is_whitespace_node(child):
                continue
            self.__code_relations.add_relation("Child", node, child, child_name)

            if prev_child is not None:
                self.__code_relations.add_relation("Sibling", prev_child, child)
            prev_child = child

        return True

    @staticmethod
    def add_ast_relations(code_rels: PythonCodeRelations, *args) -> None:
        code_rels.ast_with_metadata_wrapper.visit(AstRelations(code_rels))
