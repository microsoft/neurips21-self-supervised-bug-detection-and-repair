from typing import List, Set, Tuple

import libcst as cst
from libcst import MetadataWrapper
from libcst.metadata import CodePosition, CodeRange, PositionProvider


def is_whitespace_node(node: cst.CSTNode) -> bool:
    return isinstance(
        node, (cst.BaseParenthesizableWhitespace, cst.EmptyLine, cst.TrailingWhitespace, cst.MaybeSentinel, cst.Newline)
    )


def code_position_leq(a: CodePosition, b: CodePosition) -> bool:
    """
    Are is the position `a` before (less or equal) `b`.
    """
    if a.line < b.line:
        return True
    elif a.line == b.line and a.column <= b.column:
        return True
    return False


def code_position_within_range(pos: CodePosition, target_range: CodeRange) -> bool:
    """Is `pos` within `target_range`?"""
    return code_position_leq(target_range.start, pos) and code_position_leq(pos, target_range.end)


def code_ranges_overlap(a: CodeRange, b: CodeRange) -> bool:
    """Does a overlap with b? (symmetric)"""
    return (
        code_position_within_range(a.start, b)
        or code_position_within_range(a.end, b)
        or code_position_within_range(b.start, a)
    )


def subsumes_code_range(smaller: CodeRange, bigger: CodeRange) -> bool:
    """
    Is the `smaller` node_range entirely within the `bigger` node_range?
    """
    return code_position_within_range(smaller.start, bigger) and code_position_within_range(smaller.end, bigger)


def relative_range(base: CodeRange, target: CodeRange) -> CodeRange:
    """Return the range, relative to the base range."""
    assert subsumes_code_range(target, base), "Target range not inside base range."
    relative_start_line_no = target.start.line - base.start.line + 1
    if relative_start_line_no == 1:
        relative_start_col_no = target.start.column - base.start.column
    else:
        relative_start_col_no = target.start.column

    relative_end_line_no = relative_start_line_no + target.end.line - target.start.line
    if relative_end_line_no == 1:
        relative_end_col_no = target.end.column - base.start.column
    else:
        relative_end_col_no = target.end.column

    return CodeRange((relative_start_line_no, relative_start_col_no), (relative_end_line_no, relative_end_col_no))


class PersistentMetadataWrapper(MetadataWrapper):
    """The original MetadataWrapper keeps the metadata per-visit.

    This class makes this persistent across multiple visits.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata_cache = {}

    def resolve(self, provider):
        if provider in self._metadata_cache:
            return self._metadata_cache[provider]
        else:
            out = super().resolve(provider)
            self._metadata_cache[provider] = out
            return out


class PositionFilter(cst.CSTVisitor):
    """Collect all nodes that are within the given target node_range."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, target_range: CodeRange):
        super().__init__()
        self.__target_location = target_range
        self.nodes_within_range: Set[cst.CSTNode] = set()

    def on_visit(self, node: cst.CSTNode) -> bool:
        pos: CodeRange = self.get_metadata(PositionProvider, node)

        if subsumes_code_range(pos, self.__target_location):
            self.nodes_within_range.add(node)

        return code_ranges_overlap(pos, self.__target_location)


class AllFunctionFinder(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self):
        super().__init__()
        self.all_function_nodes: List[Tuple[cst.CSTNode, CodeRange]] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        pos: CodeRange = self.get_metadata(PositionProvider, node)
        self.all_function_nodes.append((node, pos))
        return False
