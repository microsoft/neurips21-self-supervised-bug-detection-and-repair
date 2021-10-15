import io
import math
from collections import defaultdict
from itertools import count
from typing import Dict, Iterable, List, Set, Tuple

from libcst.metadata import CodePosition, CodeRange

from buglab.utils.cstutils import relative_range


def get_text_in_range(text: str, range: CodeRange) -> str:
    with io.StringIO(text) as input_sb, io.StringIO() as output_sb:
        for line_no in count(start=1):
            next_input_line = input_sb.readline()
            if len(next_input_line) == 0:
                break  # reached EOF

            if range.start.line <= line_no <= range.end.line:
                if range.start.line == range.end.line:
                    output_sb.write(next_input_line[range.start.column : range.end.column])
                elif line_no == range.start.line:
                    output_sb.write(next_input_line[range.start.column :])
                elif line_no == range.end.line:
                    output_sb.write(next_input_line[: range.end.column])
                    break
                else:
                    output_sb.write(next_input_line)
            elif line_no > range.end.line:
                break
        return output_sb.getvalue()


def text_to_range_segments(
    text_segment: str, segment_range: CodeRange, target_ranges: Iterable[CodeRange]
) -> List[Tuple[str, Set[CodeRange]]]:
    """
    Return the text segments separated by their target ranges.

    :param text_segment: The original text
    :param segment_range: the range of the text_segment
    :param target_ranges: the target_ranges
    :return:
    """

    def non_empty(code_range: CodeRange):
        if code_range.start == code_range.end:
            return CodeRange(code_range.start, CodePosition(code_range.end.line, code_range.end.column + 1))
        return code_range

    relative_target_ranges: Dict[CodeRange, CodeRange] = {
        non_empty(relative_range(segment_range, t)): t for t in target_ranges
    }

    # Split ranges
    # First get all points
    point_to_ranges = defaultdict(set)
    for trange in relative_target_ranges:
        point_to_ranges[trange.start].add(trange)
        point_to_ranges[trange.end].add(trange)

    active_ranges = set()
    ranges: List[Tuple[str, Set[CodeRange]]] = []
    current_pos = CodePosition(0, 0)

    for target_pos in sorted(point_to_ranges, key=lambda p: (p.line, p.column)):
        ranges.append(
            (
                get_text_in_range(text_segment, CodeRange(current_pos, target_pos)),
                {relative_target_ranges[r] for r in active_ranges},
            )
        )

        relevant_ranges = point_to_ranges[target_pos]
        ranges_to_remove = active_ranges & relevant_ranges
        ranges_to_add = relevant_ranges - ranges_to_remove
        active_ranges = (active_ranges - ranges_to_remove) | ranges_to_add

        current_pos = target_pos

    assert len(active_ranges) == 0
    ranges.append((get_text_in_range(text_segment, CodeRange(current_pos, CodePosition(math.inf, math.inf))), set()))

    return ranges
