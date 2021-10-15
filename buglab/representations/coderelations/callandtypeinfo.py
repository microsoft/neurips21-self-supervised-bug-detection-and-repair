import logging
from typing import List, Optional, Sequence

import jedi
import libcst as cst
from jedi.api.classes import Name, Signature
from jedi.api.environment import Environment
from libcst.metadata import CodeRange, PositionProvider, QualifiedNameProvider, QualifiedNameSource

from buglab.representations.codereprs import PythonCodeRelations

LOGGER = logging.getLogger(__name__)


class CallAndTypeInfo(cst.CSTVisitor):
    """
    Add information about formal <-> actual arguments and types.

    Note that if multiple instances of this work in the same computer, set a different `jedi.settings.cache_directory`
    """

    METADATA_DEPENDENCIES = (PositionProvider, QualifiedNameProvider)

    def __init__(self, code_relations: PythonCodeRelations, environment: Optional[Environment]):
        super().__init__()
        self.__script = jedi.Script(
            code=code_relations.code_text, path=str(code_relations.path), environment=environment
        )
        self.__code_rels = code_relations

    def visit_Call(self, node: cst.Call):
        pos: CodeRange = self.get_metadata(PositionProvider, node.whitespace_after_func)
        signatures: Sequence[Signature] = self.__script.get_signatures(line=pos.end.line, column=pos.end.column + 1)
        for candidate_signature in signatures:
            name = candidate_signature.full_name or candidate_signature.name
            self.__code_rels.add_relation("CandidateCall", node, metadata=name)

            docstring = candidate_signature.docstring(raw=True)
            if len(docstring) > 0:
                self.__code_rels.add_relation("CandidateCallDoc", node, metadata=docstring)

        for i, arg in enumerate(node.args):
            arg_pos: CodeRange = self.get_metadata(PositionProvider, arg)
            signatures: Sequence[Signature] = self.__script.get_signatures(
                line=arg_pos.start.line, column=arg_pos.start.column + 1
            )
            for candidate_signature in signatures:
                if candidate_signature.index is None:
                    continue  # Something went wrong.
                formal_param = candidate_signature.params[candidate_signature.index]
                self.__code_rels.add_relation("MayFormalName", arg, metadata=formal_param.name)

    #    def visit_Name(self, node: cst.Name):
    #        qualified_names = self.get_metadata(QualifiedNameProvider, node)
    #        if not any(q.source == QualifiedNameSource.LOCAL for q in qualified_names):
    #            return
    #
    #        pos: CodeRange = self.get_metadata(PositionProvider, node)
    #        try:
    #            inferred_names: List[Name] = self.__script.infer(line=pos.start.line, column=pos.start.column)
    #        except Exception as e:
    #            LOGGER.exception("Error inferring type at %s for %s", pos, node.value, exc_info=e)
    #            return
    #
    #        if len(inferred_names) == 0:
    #            return
    #
    #        for qname in qualified_names:
    #            if qname.source != QualifiedNameSource.LOCAL:
    #                continue
    #            for candidate_name in inferred_names:
    #                self.__code_rels.add_relation(
    #                    "PossibleType", node, metadata=candidate_name.full_name or candidate_name.name
    #                )

    @staticmethod
    def compute_call_and_type_info(code_rels: PythonCodeRelations, environment: Optional[Environment] = None) -> None:
        callinfo = CallAndTypeInfo(code_rels, environment)
        code_rels.ast_with_metadata_wrapper.visit(callinfo)
