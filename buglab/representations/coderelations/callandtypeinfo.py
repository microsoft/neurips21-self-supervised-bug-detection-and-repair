from typing import List, Optional, Sequence

import jedi
import libcst as cst
import logging
from collections import defaultdict
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

        for arg in node.args:
            arg_pos: CodeRange = self.get_metadata(PositionProvider, arg)
            signatures: Sequence[Signature] = self.__script.get_signatures(
                line=arg_pos.start.line, column=arg_pos.start.column + 1
            )
            for candidate_signature in signatures:
                if candidate_signature.index is None:
                    continue  # Something went wrong.
                formal_param = candidate_signature.params[candidate_signature.index]
                self.__code_rels.add_relation("MayFormalName", arg, metadata=formal_param.name)

        ## Hyperedges
        if len(signatures) > 0:
            fn_name = signatures[0].name
            fn_docstring = signatures[0].docstring(raw=True)
            arguments = defaultdict(set)
            if not isinstance(node.func, cst.Name):
                arguments["self"].add(node.func)

            for i, arg in enumerate(node.args):
                if arg.keyword is not None:
                    # Keyworded arguments are easy
                    arguments[arg.keyword.value] = arg.value
                    continue

                # Since this is not a keyword argument, try to find its formal name.
                arg_pos: CodeRange = self.get_metadata(PositionProvider, arg)
                signatures: Sequence[Signature] = self.__script.get_signatures(
                    line=arg_pos.start.line, column=arg_pos.start.column + 1
                )
                potential_names = set()
                for candidate_signature in signatures:
                    if candidate_signature.index is None:
                        continue
                    formal_param = candidate_signature.params[candidate_signature.index]
                    potential_names.add(formal_param.name)

                if len(potential_names) == 0:
                    formal_name = f"$arg{i}"
                else:
                    # Pick one, which could of course be wrong.
                    formal_name = potential_names.pop()

                if arg.star == "":
                    arg_node = arg.value
                else:
                    arg_node = arg

                arguments[formal_name].add(arg_node)

            self.__code_rels.add_hyperedge_relation(
                fn_name=fn_name,
                fn_docstring=fn_docstring,
                arguments=arguments,
                return_value=node,
            )
        else:
            # No signature could be inferred, use anonymous formal args
            arguments = {
                "self": node.func,  # This isn't always here. But it's a form of an argument in many cases
            }
            unnamed_args = []
            for arg in node.args:
                if arg.keyword is not None:
                    arguments[arg.keyword.value] = arg.value
                elif arg.star == "*":
                    arguments["*args"] = arg.value
                elif arg.star == "**":
                    arguments["**kwargs"] = arg.value
                else:
                    arguments[f"$arg{len(unnamed_args)}"] = arg.value
                    unnamed_args.append(arg.value)

            if isinstance(node.func, cst.Attribute):
                fn_name = node.func.attr.value
            else:
                fn_name = "__call__"

            self.__code_rels.add_hyperedge_relation(
                fn_name=fn_name, fn_docstring=None, arguments=arguments, return_value=node
            )

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
