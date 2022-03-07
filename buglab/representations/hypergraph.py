from typing import Callable, Dict, List, Set, Tuple, Union

from collections import defaultdict

from buglab.representations.data import BugLabData, BugLabGraph, TypeAnnotationData
from buglab.utils.msgpackutils import load_msgpack_l_gz

REDUNDANT_EDGES = frozenset(
    {
        "Child",
        "Sibling",  # In AST hyperedges
        "MayFormalName",
        "CandidateCallDoc",
        "CandidateCall",  # In function call hyperedges
        "AssignedFrom",  # In syntactic hyperedges (___assign and variants)
    }
)


def _get_one_to_many(edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    aggregated_relationship = defaultdict(list)
    for k, v in edges:
        aggregated_relationship[k].append(v)
    return aggregated_relationship


def _get_many_to_one(edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    aggregated_relationship = defaultdict(list)
    for k, v in edges:
        aggregated_relationship[v].append(k)
    return aggregated_relationship


def _as_many_to_many(rels: Dict[int, List[int]]) -> Dict[Set[int], Set[int]]:
    """Convert to a many-to-many when the ins-outs are identical"""
    aggregated_relationship = defaultdict(set)
    for k, v in rels.items():
        aggregated_relationship[frozenset(v)].add(k)
    return aggregated_relationship


def convert_buglab_sample_to_hypergraph(
    datapoint: BugLabData,
    token_sequence_as_one_hyperedge: bool = True,
    chunk_size: int = 255,
) -> BugLabData:
    new_graph, node_mapping = convert_to_hypergraph(
        datapoint["graph"], token_sequence_as_one_hyperedge, chunk_size=chunk_size
    )

    candidate_rewrite_metadata = []
    for rewrite_type, metadata in datapoint["candidate_rewrite_metadata"]:
        if isinstance(metadata, int):
            metadata = node_mapping(metadata)
        elif metadata is None:
            pass
        else:
            raise Exception("Should be unreachable.")
        candidate_rewrite_metadata.append((rewrite_type, metadata))

    converted_data = {
        "graph": new_graph,
        "candidate_rewrites": datapoint["candidate_rewrites"],
        "candidate_rewrite_metadata": candidate_rewrite_metadata,  # TODO: Needs to be adjusted
        "candidate_rewrite_ranges": datapoint["candidate_rewrite_ranges"],
        "target_fix_action_idx": datapoint["target_fix_action_idx"],
        "package_name": datapoint["package_name"],
        "package_version": datapoint["package_version"],
    }
    if "candidate_rewrite_logprobs" in datapoint:
        converted_data["candidate_rewrite_logprobs"] = datapoint["candidate_rewrite_logprobs"]
    return converted_data


def convert_type_sample_to_hypergraph(
    datapoint: TypeAnnotationData,
    token_sequence_as_one_hyperedge: bool = True,
    chunk_size: int = 256,
) -> TypeAnnotationData:
    new_graph, node_mapping = convert_to_hypergraph(
        datapoint["graph"], token_sequence_as_one_hyperedge, chunk_size=chunk_size
    )
    datapoint["graph"] = new_graph
    datapoint["annotations"] = {
        str(node_mapping(int(node_idx))): annotation for node_idx, annotation in datapoint["annotations"].items()
    }
    return datapoint


def convert_to_hypergraph(
    graph: BugLabGraph,
    token_sequence_as_one_hyperedge: bool = True,
    chunk_size: int = 256,
) -> Union[BugLabGraph, Callable[[int], int]]:
    next_siblings = dict(graph["edges"]["Sibling"]) if "Sibling" in graph["edges"] else {}
    prev_sibling = {v: k for k, v in next_siblings.items()}

    for e in REDUNDANT_EDGES:
        # These are redundant (all info in hyperedges)
        try:
            del graph["edges"][e]
        except KeyError:
            pass

    convert_dataflow_edges(graph)
    convert_control_flow_edges(graph)
    convert_token_sequence(graph, token_sequence_as_one_hyperedge, chunk_size=chunk_size)

    # Update AST hyperedges
    def map_hyperedge(hyperedge):
        if hyperedge["name"] != "$AstChild":
            return hyperedge

        new_args = {}
        for k, v in hyperedge["args"].items():
            if isinstance(v, list) and len(v) > 1:
                # First, order the children
                children = list(filter(lambda x: x not in prev_sibling or prev_sibling[x] not in v, v))
                assert len(children) == 1
                while children[-1] in next_siblings and next_siblings[children[-1]] in v:
                    children.append(next_siblings[children[-1]])

                for i, child in enumerate(children):
                    if i == 0:
                        new_args[k] = [child]
                    else:
                        new_args[f"{k}{i}"] = [child]
                assert len(v) - 1 == i
            else:
                new_args[k] = v
        hyperedge["args"] = new_args
        return hyperedge

    graph["hyperedges"] = list(map(map_hyperedge, graph["hyperedges"]))

    assert len(graph["edges"]) == 0  # No edges should be left.

    # There will be some unused nodes (e.g. where the `MayFormalName`, `CandidateCallDoc` was pointing to).
    # These are commonly last, and no changes are needed. But enforce this here.
    # Find all unconnected nodes
    all_used_nodes = set()
    for hyperedge in graph["hyperedges"]:
        for arg, node_ids in hyperedge["args"].items():
            all_used_nodes.update(node_ids)

    unused_nodes = set(range(len(graph["nodes"]))) - all_used_nodes
    nodes = graph["nodes"]
    if len(unused_nodes) > 0:
        assert not any(n in graph["reference_nodes"] for n in unused_nodes)
        old_to_new_mapping = {}  # Map any changed node ids

        next_node_idx = min(unused_nodes)
        nodes = graph["nodes"][:next_node_idx]
        for old_node_idx in range(min(unused_nodes) + 1, len(graph["nodes"])):
            if old_node_idx in unused_nodes:
                continue
            old_to_new_mapping[old_node_idx] = next_node_idx
            next_node_idx += 1
            nodes.append(graph["nodes"][old_node_idx])

        def mapping(old_node_idx: int) -> int:
            if not unused_nodes:
                return old_node_idx
            assert old_node_idx not in unused_nodes
            # Default to the old ids unless the old_id is in the map.
            return old_to_new_mapping.get(old_node_idx, old_node_idx)

        needs_mapping = True
    else:
        mapping = lambda x: x
        needs_mapping = False

    graph["nodes"] = nodes
    if needs_mapping:
        # in-place mapping of hyperedge args
        for hyperedge in graph["hyperedges"]:
            for value in hyperedge["args"].values():
                for i in range(len(value)):
                    value[i] = mapping(value[i])

        graph["reference_nodes"] = [mapping(n) for n in graph["reference_nodes"]]

    return graph, mapping


def convert_token_sequence(graph, token_sequence_as_one_hyperedge, chunk_size=256, chunk_node_prefix: str = "CHUNK"):
    if token_sequence_as_one_hyperedge:
        graph["nodes"].extend([f"[{chunk_node_prefix}-START]", f"[{chunk_node_prefix}-END]"])
        chunk_start_id, chunk_end_id = len(graph["nodes"]) - 2, len(graph["nodes"]) - 1
        next_token_edges = {f: t for f, t in graph["edges"]["NextToken"]}
        del graph["edges"]["NextToken"]

        first_token = set(next_token_edges.keys()) - set(next_token_edges.values())
        assert len(first_token) == 1
        first_token_idx = first_token.pop()

        def get_token_sequence():
            seen_tokens = {first_token_idx}
            next_token = first_token_idx
            yield next_token
            while next_token in next_token_edges:
                next_token = next_token_edges[next_token]
                if next_token in seen_tokens:
                    raise Exception("Cyclic token sequence in %s", graph["path"])
                seen_tokens.add(next_token)
                yield next_token

        token_sequence = list(get_token_sequence())

        def get_chunk_tokens(chunk):
            yield chunk_start_id
            yield from chunk
            yield chunk_end_id

        chunk_size -= 2  # Account for CHUNK-START, CHUNK-END
        for chunk_idx in range(0, len(token_sequence), chunk_size):
            # Use a special name "$posXXX" to allow for special-casing the relevant positional encodings.
            hyperedge = {
                "name": "$Tokens",
                "docstring": None,
                "args": {
                    f"$pos{chunk_idx+i}": [t]
                    for i, t in enumerate(get_chunk_tokens(token_sequence[chunk_idx : chunk_idx + chunk_size]))
                },
            }
            graph["hyperedges"].append(hyperedge)

    else:
        for from_token, to_token in graph["edges"]["NextToken"]:
            graph["hyperedges"].append(
                {"name": "$NextToken", "docstring": None, "args": {"from": [from_token], "to": [to_token]}}
            )
        del graph["edges"]["NextToken"]


def convert_dataflow_edges(graph):
    ####
    # Convert the "OccurrenceOf" and "MayFinalUseOf" into a hyperedge
    ####
    if "OccurrenceOf" in graph["edges"]:
        symbol_to_occurence = _get_many_to_one(graph["edges"]["OccurrenceOf"])
        del graph["edges"]["OccurrenceOf"]
    else:
        symbol_to_occurence = {}

    if "MayFinalUseOf" in graph["edges"]:
        symbol_to_may_final_use = _get_many_to_one(graph["edges"]["MayFinalUseOf"])
        del graph["edges"]["MayFinalUseOf"]
    else:
        symbol_to_may_final_use = {}

    for symbol_node_id, occurence_node_ids in symbol_to_occurence.items():
        # As a representation, we don't need the symbol_node (i.e. symbol argument)
        # However keeping it makes this compatible with the surrounding code (e.g. varmisuse)
        args = {"symbol": [symbol_node_id], "occurrence": occurence_node_ids}

        if symbol_node_id in symbol_to_may_final_use:
            args["may_final_use"] = symbol_to_may_final_use[symbol_node_id]

        graph["hyperedges"].append({"name": "$SymbolInfo", "docstring": None, "args": args})

    # there were occasions where the two keys below were not found
    for original_edge_name, hedge_name in (("LastMayWrite", "$MayWrite"), ("NextMayUse", "$MayUse")):
        if original_edge_name not in graph["edges"]:
            continue
        prev_nodes_to_next_nodes = _as_many_to_many(_get_many_to_one(graph["edges"][original_edge_name]))
        del graph["edges"][original_edge_name]

        for prev, next in prev_nodes_to_next_nodes.items():
            graph["hyperedges"].append(
                {
                    "name": hedge_name,
                    "docstring": None,
                    "args": {"last_may_writes": list(prev), "uses": list(next)}
                    if original_edge_name == "NextMayUse"
                    else {"previous_uses": list(prev), "next_uses": list(next)},
                }
            )


def convert_control_flow_edges(graph):
    ####
    # Convert the "ReturnsFrom" and "YieldsFrom" into a hyperedge
    ####
    if "ReturnsFrom" in graph["edges"]:
        returns_fn_to_node = _get_one_to_many(graph["edges"]["ReturnsFrom"])
        del graph["edges"]["ReturnsFrom"]
        for fn_node_id, return_statement_node_ids in returns_fn_to_node.items():
            graph["hyperedges"].append(
                {"name": "$Returns", "docstring": None, "args": {"fn": [fn_node_id], "from": return_statement_node_ids}}
            )

    if "YieldsFrom" in graph["edges"]:
        yields_fn_to_node = _get_one_to_many(graph["edges"]["YieldsFrom"])
        del graph["edges"]["YieldsFrom"]
        for fn_node_id, yields_statement_node_ids in yields_fn_to_node.items():
            graph["hyperedges"].append(
                {"name": "$Yields", "docstring": None, "args": {"fn": [fn_node_id], "from": yields_statement_node_ids}}
            )

    ####
    # Convert the "ControlFlowNext" into a hyperedge. This merges previous states
    ####
    prev_nodes_to_next_nodes = _as_many_to_many(_get_many_to_one(graph["edges"]["ControlFlowNext"]))
    del graph["edges"]["ControlFlowNext"]
    for prev_states, next_states in prev_nodes_to_next_nodes.items():
        graph["hyperedges"].append(
            {
                "name": "$ControlFlowNext",
                "docstring": None,
                "args": {"next": list(next_states), "prev": list(prev_states)},
            }
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert graphs to hypergraphs and print as .dot files")
    parser.add_argument(
        "input_msgpack",
        type=str,
        help="the path to the input file",
    )
    parser.add_argument("output_path", type=str, help="the path to the target .dot file")
    parser.add_argument("idx", type=int, help="the index of the target graph.")
    parser.add_argument("--next-token-edge", help="use next token edge.", action="store_true")

    args = parser.parse_args()

    for i, data in enumerate(load_msgpack_l_gz(args.input_msgpack)):
        if i == args.idx:
            data = convert_buglab_sample_to_hypergraph(data, token_sequence_as_one_hyperedge=not args.next_token_edge)
            graph = data["graph"]

            code_range = graph["code_range"]
            print(
                f"Creating .dot for {graph['path']} (package: {data['package_name']}=={data['package_version']})"
                f" @ ({code_range[0][0]},{code_range[0][1]})--({code_range[1][0]},{code_range[1][1]}) "
            )
            BugLabGraph.to_dot(graph, args.output_path, {})
            break
    else:
        print("The target graph was not found.")
