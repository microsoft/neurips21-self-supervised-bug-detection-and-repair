import re
from os import PathLike
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from dpu_utils.codeutils import split_identifier_into_parts
from dpu_utils.mlutils import Vocabulary
from ptgnn.neuralmodels.gnn import GraphData
from typing_extensions import TypedDict

from buglab.utils.msgpackutils import load_msgpack_l_gz


class BugLabGraph(TypedDict):
    nodes: List[str]
    edges: Dict[str, List[Union[Tuple[int, int], Tuple[int, int, str]]]]
    path: str
    text: str
    reference_nodes: List[int]
    code_range: Tuple[Tuple[int, int], Tuple[int, int]]

    @classmethod
    def to_dot(cls, data: "BugLabGraph", filepath: PathLike, edge_colors: Optional[Dict[str, str]]) -> None:
        def escape(string: str) -> str:
            return string.replace('"', '\\"')

        if edge_colors is None:
            edge_colors = {}

        reference_nodes = set(data["reference_nodes"])
        colored_nodes = [
            ("red", {target_idx for _, target_idx in data["edges"]["PossibleType"]}),
            ("orange", {target_idx for _, target_idx in data["edges"]["OccurrenceOf"]}),
            ("blue", {target_idx for _, target_idx in data["edges"]["CandidateCall"]}),
            ("yellow", {target_idx for _, target_idx in data["edges"]["CandidateCallDoc"]}),
        ]

        token_node_idxs = {target_idx for _, target_idx in data["edges"]["NextToken"]}

        with open(filepath, "w") as f:
            f.write("digraph {\n\tcompound=true;")

            for node_idx, node_lbl in enumerate(data["nodes"]):
                if node_idx in token_node_idxs:
                    continue
                node_specs = f'shape="rectangle", label="{node_idx}: {escape(node_lbl)}"'
                for color, node_set in colored_nodes:
                    if node_idx in node_set:
                        node_specs += f', style=filled, fillcolor="{color}"'
                        break

                if node_idx in reference_nodes:
                    node_specs += ", style=striped"

                f.write(f"\t node{node_idx}[{node_specs}];\n")

            f.write("subgraph clustertokens {\nrankdir=TB;rank=min;\n")
            for token_idx in token_node_idxs:
                node_lbl = data["nodes"][token_idx]
                node_specs = f'shape="rectangle", label="{token_idx}: {escape(node_lbl)}"'
                if token_idx in reference_nodes:
                    node_specs += ", style=striped"

                f.write(f"\t node{token_idx}[{node_specs}];\n")
            f.write("}")  # subgraph

            for edge_name, edges in data["edges"].items():
                edge_color = edge_colors.get(edge_name, "black")
                for edge in edges:
                    if len(edge) == 2:
                        from_idx, to_idx = edge
                        label = None
                    else:
                        from_idx, to_idx, label = edge

                    edge_style = f'color="{edge_color}"'

                    if edge_name == "NextToken":
                        edge_style += ", headport=n, tailport=s, weight=500"
                    elif edge_name == "Child":
                        edge_style += ", headport=e, tailport=w, weight=500"
                    elif edge_name == "Sibling":
                        edge_style += ", style=dashed"

                    if label is None:
                        label = edge_name
                    else:
                        label = f"{edge_name} ({label})"

                    f.write(f'\tnode{from_idx} -> node{to_idx} [xlabel="{label}", {edge_style}];\n')
            f.write("}\n")  # graph


IS_IDENTIFIER = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def add_open_vocab_nodes_and_edges(graph: BugLabGraph) -> None:
    token_nodes = set()
    if "NextToken" not in graph["edges"]:
        return
    for n1, n2 in graph["edges"]["NextToken"]:
        token_nodes.add(n1)
        token_nodes.add(n2)

    vocab_nodes: Dict[str, int] = {}
    vocab_edges: List[Tuple[int, int]] = []

    all_nodes = graph["nodes"]
    for node_idx in token_nodes:
        token_str = all_nodes[node_idx]
        if not IS_IDENTIFIER.match(token_str):
            continue
        for subtoken in split_identifier_into_parts(token_str):
            subtoken_node_idx = vocab_nodes.get(subtoken)
            if subtoken_node_idx is None:
                subtoken_node_idx = len(all_nodes)
                all_nodes.append(subtoken)
                vocab_nodes[subtoken] = subtoken_node_idx
            vocab_edges.append((node_idx, subtoken_node_idx))

    graph["edges"]["HasSubtoken"] = vocab_edges


def _as_np_array(arr):
    if len(arr) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(arr, dtype=np.int32)


class BugLabData(TypedDict):
    graph: BugLabGraph
    candidate_rewrites: List[Tuple[str, Any]]
    candidate_rewrite_metadata: List[Tuple[str, Any]]
    candidate_rewrite_ranges: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    target_fix_action_idx: Optional[int]
    package_name: str
    candidate_rewrite_logprobs: Optional[List[float]]

    @classmethod
    def as_graph_data(cls, data: "BugLabData") -> Tuple[GraphData, Optional[int]]:
        candidate_node_idxs, inv = np.unique(data["graph"]["reference_nodes"], return_inverse=True)
        if data["target_fix_action_idx"] is not None:
            target_node_idx = inv[data["target_fix_action_idx"]]
            assert (
                data["graph"]["reference_nodes"][data["target_fix_action_idx"]] == candidate_node_idxs[target_node_idx]
            )
        else:
            target_node_idx = None

        add_open_vocab_nodes_and_edges(data["graph"])
        return (
            GraphData(
                node_information=data["graph"]["nodes"],
                edges={
                    e_type: _as_np_array([(e[0], e[1]) for e in adj_list])
                    for e_type, adj_list in data["graph"]["edges"].items()
                },
                edge_features={
                    e_type: [e[2] if len(e) >= 3 else Vocabulary.get_pad() for e in adj_list]
                    for e_type, adj_list in data["graph"]["edges"].items()
                },
                reference_nodes={
                    "candidate_nodes": candidate_node_idxs,
                },
            ),
            target_node_idx,
        )


class TypeAnnotationPrediction(NamedTuple):
    score: float
    annotation: str


class TypeAnnotationData(TypedDict):
    graph: BugLabGraph
    annotations: Dict[str, str]  # index of node -> type annotation
    package_name: str

    @classmethod
    def as_graph_data(cls, data: "TypeAnnotationData") -> GraphData:
        candidate_node_idxs, inv = np.unique(data["graph"]["reference_nodes"], return_inverse=True)

        add_open_vocab_nodes_and_edges(data["graph"])

        return GraphData(
            node_information=data["graph"]["nodes"],
            edges={
                e_type: _as_np_array([(e[0], e[1]) for e in adj_list])
                for e_type, adj_list in data["graph"]["edges"].items()
            },
            reference_nodes={"candidate_nodes": candidate_node_idxs},
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print graphs as .dot files")
    parser.add_argument(
        "input_msgpack",
        type=str,
        help="the path to the input file",
    )

    parser.add_argument("output_path", type=str, help="the path to the target .dot file")

    parser.add_argument("--idx", type=int, help="the index of the target graph.")

    # TODO: Add mutually exclusive options (path, range filters)
    args = parser.parse_args()

    EDGE_COLORS = {
        "LastMayWrite": "red",
        "NextMayUse": "blue",
        "OccurrenceOf": "green",
        "ComputedFrom": "brown",
        "NextToken": "hotpink",
        "Sibling": "lightgray",
    }

    for i, data in enumerate(load_msgpack_l_gz(args.input_msgpack)):
        if i == args.idx:
            graph: BugLabGraph = data["graph"]
            code_range = graph["code_range"]
            print(
                f"Creating .dot for {graph['path']} (package: {data['package_name']}=={data['package_version']})"
                f" @ ({code_range[0][0]},{code_range[0][1]})--({code_range[1][0]},{code_range[1][1]}) "
            )
            BugLabGraph.to_dot(graph, args.output_path, EDGE_COLORS)
            break
    else:
        print("The target graph was not found.")
