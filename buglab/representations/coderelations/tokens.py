import io
import tokenize
from collections import OrderedDict

import libcst as cst
from libcst.metadata import CodePosition, CodeRange, PositionProvider

from buglab.representations.codereprs import DummyEntity, PythonCodeRelations
from buglab.utils.cstutils import code_position_leq, is_whitespace_node, subsumes_code_range

__all__ = ["TokenRelations"]


class TokenRelations(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, code_relations: PythonCodeRelations):
        super().__init__()
        self.__code_relations = code_relations

        self.__tokens_to_visit = OrderedDict()
        self.__token_nodes = OrderedDict()
        with io.BytesIO(code_relations.code_text.encode()) as token_stream:
            tokens_iter = tokenize.tokenize(token_stream.readline)
            for toknum, tokval, (srow, scol), (erow, ecol), _ in tokens_iter:
                if toknum in {
                    tokenize.ENCODING,
                    tokenize.NEWLINE,
                    tokenize.ERRORTOKEN,
                    tokenize.NL,
                    tokenize.N_TOKENS,
                    tokenize.NT_OFFSET,
                    tokenize.ENDMARKER,
                }:
                    continue
                elif toknum == tokenize.INDENT:
                    tokval = "<INDENT>"
                elif toknum == tokenize.DEDENT:
                    tokval = "<DEDENT>"
                self.__tokens_to_visit[(srow, scol), (erow, ecol)] = tokval
                self.__token_nodes[(srow, scol), (erow, ecol)] = tokval
        self.__prev_token = None

    def add_edges(self) -> None:
        assert len(self.__tokens_to_visit) == 0
        for pos, tokval in self.__token_nodes.items():
            if self.__prev_token is not None:
                self.__code_relations.add_relation("NextToken", self.__prev_token, tokval)
            self.__prev_token = tokval

    def on_leave(self, original_node: cst.CSTNode) -> None:
        """Do a post-order visit and gradually "consume" tokens."""
        pos: CodeRange = self.get_metadata(PositionProvider, original_node)

        if is_whitespace_node(original_node):
            return

        token_pos_of_node = []
        for token_pos in self.__tokens_to_visit:
            if subsumes_code_range(CodeRange(*token_pos), pos):
                token_pos_of_node.append(token_pos)
            if code_position_leq(pos.end, CodePosition(*token_pos[0])):
                break  # Tokens appear in order. We can safely stop here.

        if (
            len(token_pos_of_node) == 1
            and pos.start.line == token_pos_of_node[0][0][0]
            and pos.end.line == token_pos_of_node[0][1][0]
            and pos.start.column == token_pos_of_node[0][0][1]
            and pos.end.column == token_pos_of_node[0][1][1]
        ):
            # This node represents a token
            self.__token_nodes[token_pos_of_node[0]] = original_node
        else:
            for t in token_pos_of_node:
                self.__token_nodes[t] = DummyEntity(self.__tokens_to_visit[t], CodeRange(*t))
                self.__code_relations.add_relation("Child", original_node, self.__token_nodes[t])

        for t in token_pos_of_node:
            del self.__tokens_to_visit[t]

    @staticmethod
    def add_token_relations(code_rels: PythonCodeRelations, *args) -> None:
        visitor = TokenRelations(code_rels)
        code_rels.ast_with_metadata_wrapper.visit(visitor)
        visitor.add_edges()
