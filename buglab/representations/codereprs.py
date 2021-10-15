from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import libcst as cst
from libcst.metadata import CodeRange
from typing_extensions import Final

from buglab.representations.data import BugLabGraph
from buglab.utils.cstutils import PersistentMetadataWrapper, PositionFilter, subsumes_code_range
from buglab.utils.text import get_text_in_range


class DummyEntity:
    __slots__ = ("value", "range")

    def __init__(self, value: str, node_range: CodeRange):
        self.value: Final = value
        self.range: Final = node_range


TNode = Union[cst.CSTNode, DummyEntity]


class PythonCodeRelations:
    def __init__(self, code_text: str, path: Path):
        self.__code_text = code_text
        self.__path: Final = path

        self.__ast_with_metadata_wrapper = PersistentMetadataWrapper(cst.parse_module(code_text), unsafe_skip_copy=True)

        self.__relations: DefaultDict[str, DefaultDict[TNode, Set[Tuple[TNode, Any]]]] = defaultdict(
            lambda: defaultdict(set)
        )

    @property
    def ast(self) -> cst.Module:
        return self.__ast_with_metadata_wrapper.module

    @property
    def ast_with_metadata_wrapper(self) -> cst.MetadataWrapper:
        return self.__ast_with_metadata_wrapper

    @property
    def code_text(self) -> str:
        return self.__code_text

    @property
    def path(self) -> Path:
        return self.__path

    def add_relation(
        self, relation_kind: str, from_node: TNode, to_node: Optional[TNode] = None, metadata: Any = None
    ) -> None:
        assert from_node is not None
        node_relations = self.__relations[relation_kind][from_node]
        assert to_node is not None or metadata is not None
        node_relations.add((to_node, metadata))

    def __map_nodes_to_idx(self, target_range: Optional[CodeRange]) -> Dict[Any, int]:
        """Map all nodes within the target range to a unique id."""
        if target_range is not None:
            nodes_to_use = self.__get_nodes_within_range(target_range)
        else:
            nodes_to_use = None

        node_to_idx: Dict[Any, int] = {}

        def add_node(node):
            if node not in node_to_idx:
                node_to_idx[node] = len(node_to_idx)

        for node_rels in self.__relations.values():
            for from_node, target_nodes in node_rels.items():
                if nodes_to_use is not None and isinstance(from_node, cst.CSTNode) and from_node not in nodes_to_use:
                    continue
                elif (
                    nodes_to_use is not None
                    and isinstance(from_node, DummyEntity)
                    and not subsumes_code_range(from_node.range, target_range)
                ):
                    continue
                add_node(from_node)
                for node, metadata in target_nodes:
                    if isinstance(node, cst.CSTNode) and (nodes_to_use is None or node in nodes_to_use):
                        add_node(node)
                    elif isinstance(node, DummyEntity) and (
                        target_range is None or subsumes_code_range(node.range, target_range)
                    ):
                        add_node(node)
                    elif node is None:
                        assert metadata is not None, "Both the node and the metadata is empty."
                        add_node(metadata)

        return node_to_idx

    def __get_nodes_within_range(self, range: CodeRange) -> Set[cst.CSTNode]:
        visitor = PositionFilter(range)
        self.__ast_with_metadata_wrapper.visit(visitor)
        return visitor.nodes_within_range

    def __node_to_label(self, node: Union[cst.CSTNode, DummyEntity]) -> str:
        if isinstance(node, cst.Name):
            return node.value
        elif isinstance(node, cst.Integer):
            return str(node.value)
        elif isinstance(node, cst.Float):
            return str(node.value)
        elif isinstance(node, cst.SimpleString):
            return node.value
        elif isinstance(node, cst.Imaginary):
            return node.value
        elif isinstance(node, DummyEntity):
            return node.value
        elif hasattr(node, "_get_token"):
            return node._get_token()

        return type(node).__name__

    def as_serializable(
        self,
        target_range: Optional[CodeRange] = None,
        reference_nodes: List[cst.CSTNode] = None,
    ) -> Tuple[BugLabGraph, Dict[Any, int]]:
        # TODO: Refactor this out of this class to allow multiple different serializable representations.
        node_idxs = self.__map_nodes_to_idx(target_range)

        all_nodes = [None] * len(node_idxs)
        for node, idx in node_idxs.items():
            assert all_nodes[idx] is None
            if isinstance(node, (cst.CSTNode, DummyEntity)):
                node_lbl = self.__node_to_label(node)
                all_nodes[idx] = node_lbl
            else:
                all_nodes[idx] = self.get_abbrv_symbol_name(node)

        edges = {}
        for rel_type, nodes in self.__relations.items():
            edges_for_rel = []
            edges[rel_type] = edges_for_rel
            for from_node, to_nodes in nodes.items():
                from_idx = node_idxs.get(from_node)
                if from_idx is None:
                    continue

                for node, metadata in to_nodes:
                    if node is not None and node not in node_idxs:
                        continue
                    to_idx = node_idxs[node if node is not None else metadata]
                    if metadata is None or node is None:
                        edges_for_rel.append((from_idx, to_idx))
                    else:
                        edges_for_rel.append((from_idx, to_idx, metadata))

        data = {
            "nodes": all_nodes,
            "edges": edges,
            "path": str(self.__path),
            "reference_nodes": [node_idxs.get(n) for n in reference_nodes],
            "text": get_text_in_range(self.__code_text, target_range),
        }
        if target_range is not None:
            data["code_range"] = (
                (target_range.start.line, target_range.start.column),
                (target_range.end.line, target_range.end.column),
            )
        return data, node_idxs

    def get_abbrv_symbol_name(self, node: str):
        local_s = node.rfind("<locals>.")
        if local_s >= 0:
            node = node[local_s + len("<locals>.") :]
        return node

    def as_dot(
        self, filepath: Path, edge_colors: Optional[Dict[str, str]] = None, target_range: Optional[CodeRange] = None
    ):
        node_idxs = self.__map_nodes_to_idx(target_range)

        def escape(string: str) -> str:
            return string.replace('"', '\\"')

        if edge_colors is None:
            edge_colors = {}

        token_nodes = set()
        for from_node, to_nodes in self.__relations["NextToken"].items():
            token_nodes.update(t for t, m in to_nodes)
            token_nodes.add(from_node)

        with open(filepath, "w") as f:
            f.write("digraph {\n\tcompound=true;\n")
            for node, node_idx in node_idxs.items():
                if node in token_nodes:
                    continue
                if isinstance(node, (cst.CSTNode, DummyEntity)):
                    node_lbl = escape(self.__node_to_label(node))
                    f.write(f'\t node{node_idx}[shape="rectangle", label="{node_lbl}"];\n')
                else:
                    node_lbl = escape(node)
                    f.write(
                        f'\t node{node_idx}[shape="rectangle", label="{node_lbl}", style=filled, fillcolor="orange"];\n'
                    )
            f.write('\tsubgraph clusterNextToken {\n\tlabel="Tokens";\n\trank="same";\n')
            for token_node in token_nodes:
                if token_node not in node_idxs:
                    continue
                node_lbl = escape(self.__node_to_label(token_node))
                f.write(f'\t\tnode{node_idxs[token_node]}[shape="rectangle", label="{node_lbl}"];\n')
            edge_color = edge_colors.get("NextToken", "black")
            self.__create_dot_edges(edge_color, f, node_idxs, self.__relations["NextToken"], "NextToken")
            f.write("\t}\n")  # subgraph

            for rel_type, nodes in self.__relations.items():
                if rel_type == "NextToken":
                    continue
                edge_color = edge_colors.get(rel_type, "black")
                self.__create_dot_edges(edge_color, f, node_idxs, nodes, rel_type)
            f.write("}\n")  # graph

    def __create_dot_edges(self, edge_color, f, node_idxs, nodes, rel_type, indent="\t", weight=None):
        for from_node, to_nodes in nodes.items():
            from_idx = node_idxs.get(from_node)
            if from_idx is None:
                continue

            edge_style = f'color="{edge_color}", splines=ortho'
            if weight:
                edge_style += f", weight={weight}"

            for node, metadata in to_nodes:
                if node is not None and node not in node_idxs:
                    continue
                to_idx = node_idxs[node if node is not None else metadata]
                if metadata is None or node is None:
                    f.write(f'{indent}node{from_idx} -> node{to_idx} [label="{rel_type}" {edge_style}];\n')
                else:
                    f.write(f'{indent}node{from_idx} -> node{to_idx} [label="{rel_type}.{metadata}" {edge_style}];\n')
