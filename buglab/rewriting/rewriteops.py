import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Tuple, TypeVar

import libcst as cst
from libcst.metadata import CodeRange, PositionProvider

T = TypeVar("T")


@dataclass(frozen=True, repr=False)
class AbstractRewriteOp(ABC):
    target_range: CodeRange

    @abstractmethod
    def rewrite(self: T, code: str) -> Tuple[str, T]:
        """Apply the rewrite to the module, returning the modified code text and the revert operation."""

    @classmethod
    @abstractmethod
    def op_name(self) -> str:
        """The name of the rewrite operation."""

    @abstractmethod
    def rewrite_data(self) -> Any:
        """The data related to this rewrite operation."""

    @abstractmethod
    def _operands_repr(self) -> str:
        """The string representation of the operands."""

    def __repr__(self):
        return (
            f"{self.op_name()}({self._operands_repr()} @("
            f"{self.target_range.start.line},{self.target_range.start.column})"
            f"->({self.target_range.end.line},{self.target_range.end.column})"
            ")"
        )


@dataclass(frozen=True, repr=False)
class ReplaceCodeTextOp(AbstractRewriteOp):

    replaced_text: str

    @classmethod
    def op_name(cls) -> str:
        return "ReplaceText"

    def _operands_repr(self) -> str:
        return f"target={repr(self.replaced_text)}"

    def rewrite_data(self) -> str:
        return self.replaced_text

    def rewrite(self, code: str) -> Tuple[str, "ReplaceCodeTextOp"]:
        assert self.target_range.start.line == self.target_range.end.line, "Cannot replace text on multiple lines."
        with io.StringIO(code) as input_sb, io.StringIO() as output_sb:
            for line_no in count(start=1):
                next_input_line = input_sb.readline()
                if len(next_input_line) == 0:
                    break  # reached EOF

                if self.target_range.start.line == line_no:
                    output_sb.write(next_input_line[: self.target_range.start.column])
                    output_sb.write(self.replaced_text)
                    output_sb.write(next_input_line[self.target_range.end.column :])

                    revert_op = ReplaceCodeTextOp(
                        replaced_text=next_input_line[self.target_range.start.column : self.target_range.end.column],
                        target_range=CodeRange(
                            (self.target_range.start.line, self.target_range.start.column),
                            (self.target_range.start.line, self.target_range.start.column + len(self.replaced_text)),
                        ),
                    )

                else:
                    output_sb.write(next_input_line)
            return output_sb.getvalue(), revert_op


@dataclass(frozen=True, repr=False)
class ArgSwapOp(AbstractRewriteOp):
    idxs: Tuple[int, int] = field(default_factory=lambda x: (min(x[0], x[1], max(x[0], x[1]))))

    @classmethod
    def op_name(cls) -> str:
        return "ArgSwap"

    def _operands_repr(self) -> str:
        return f"idxs={self.idxs[0]}<->{self.idxs[1]}"

    def rewrite_data(self) -> Tuple[int, int]:
        return self.idxs

    class _SwapListElements(cst.CSTTransformer):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def __init__(self, target_range: CodeRange, swapped_idxs: Tuple[int, int]):
            super().__init__()
            self.target_found = False
            self.__target_range = target_range
            self.__swapped_idxs = swapped_idxs

        def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
            node_range: CodeRange = self.get_metadata(PositionProvider, original_node.func)
            if self.__target_range != node_range:
                return updated_node

            assert not self.target_found, "Target target_range has already been found."
            self.target_found = True

            idx1, idx2 = self.__swapped_idxs
            arg1 = updated_node.args[idx1]
            arg2 = updated_node.args[idx2]
            assert arg1 is not None
            assert arg2 is not None

            arg1_new = arg1.with_changes(value=arg2.value)
            arg2_new = arg2.with_changes(value=arg1.value)
            args_new = list(updated_node.args)
            args_new[idx1], args_new[idx2] = arg1_new, arg2_new

            return original_node.with_changes(args=tuple(args_new))

    def rewrite(self, code: str) -> Tuple[str, "ArgSwapOp"]:
        module = cst.parse_module(code)
        wrapper = cst.metadata.MetadataWrapper(module, unsafe_skip_copy=True)
        swap_visitor = self._SwapListElements(self.target_range, self.idxs)
        result = wrapper.visit(swap_visitor)
        assert swap_visitor.target_found, "The target target_range of the rewrite was not found."
        return result.code, self
