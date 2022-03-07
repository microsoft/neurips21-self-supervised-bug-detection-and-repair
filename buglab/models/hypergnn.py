from typing import Any, Dict, Final, Iterable, List, Optional, TypeVar, Union
from typing_extensions import Literal

import logging
import numpy as np
import torch
import torch_scatter
from collections import defaultdict
from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
from ptgnn.neuralmodels.gnn import GnnOutput
from ptgnn.neuralmodels.reduceops.varsizedsummary import (
    AbstractVarSizedElementReduce,
    ElementsToSummaryRepresentationInput,
    SelfAttentionVarSizedElementReduce,
    SimpleVarSizedElementReduce,
)
from torch import nn
from torch_scatter import scatter

from buglab.models.layers.allset_reduce import (
    AllDeepSetElementReduce,
    AllSetTransformerReduce,
    MultiheadSelfAttentionVarSizedKeysElementReduce,
)
from buglab.representations.data import HyperedgeData, HypergraphData, TensorizedHyperedgeData, TensorizedHypergraphData

LOGGER = logging.getLogger(__name__)
TNodeData = TypeVar("TNodeData")
TNeuralModule = TypeVar("TNeuralModule")
TTensorizedNodeData = TypeVar("TTensorizedNodeData")
TensorizedHyperedgeArgs = TypeVar("TensorizedHyperedgeArgs")


def _get_unique_idx(el, unq_dict):
    if el not in unq_dict:
        unq_dict[el] = len(unq_dict)
    return unq_dict[el]


class TransformerUpdateNetwork(nn.Module):
    def __init__(self, hidden_state_size: int, dropout_rate: float, feedfoward_dim: int = 2048):
        super().__init__()
        self._dropout = nn.Dropout(dropout_rate)
        self._norm1 = nn.LayerNorm(hidden_state_size)
        self._ff_block = nn.Sequential(
            nn.Linear(hidden_state_size, feedfoward_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feedfoward_dim, hidden_state_size),
            nn.Dropout(dropout_rate),
        )

        self._norm2 = nn.LayerNorm(hidden_state_size)

    def forward(self, *, inputs: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        node_aggregated = self._norm1(self._dropout(inputs) + prev_state)
        ff_out = self._ff_block(node_aggregated)

        # Enforcing the dtype is necessary since layerNorm returns float32 even in AMP.
        return self._norm2(node_aggregated + ff_out).to(dtype=prev_state.dtype)


class NoNodeUpdateNetwork(nn.Module):
    def forward(self, *, inputs: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        return inputs


class AddAndNorm(nn.Module):
    def __init__(self, hidden_state_size: int, dropout_rate: float):
        super().__init__()
        self._norm1 = nn.LayerNorm(hidden_state_size)
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, *, inputs: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        return self._norm1(self._dropout(inputs) + prev_state).to(prev_state.dtype)


class LinearUpdateNetwork(nn.Module):
    def __init__(self, hidden_state_size: int, dropout_rate: float, add_residual: bool):
        super().__init__()
        self._net = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_state_size, hidden_state_size),
        )
        self._add_residual = add_residual

    def forward(self, *, inputs: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        if self._add_residual:
            return self._net(inputs) + prev_state
        return self._net(inputs)


class PnaLikeVarSizedElementReduce(AbstractVarSizedElementReduce):
    def __init__(self, hidden_state_size: int):
        super().__init__()
        self._map_to_size = nn.Linear(hidden_state_size * 4, hidden_state_size)

    def forward(self, inputs: ElementsToSummaryRepresentationInput) -> torch.Tensor:
        sum_reduce = scatter(
            src=inputs.element_embeddings,
            index=inputs.element_to_sample_map,
            dim=0,
            dim_size=inputs.num_samples,
            reduce="sum",
        )
        mean_reduce = scatter(
            src=inputs.element_embeddings,
            index=inputs.element_to_sample_map,
            dim=0,
            dim_size=inputs.num_samples,
            reduce="mean",
        )
        maxmin_reduce = scatter(
            src=torch.cat([inputs.element_embeddings, -inputs.element_embeddings], dim=-1),
            index=inputs.element_to_sample_map,
            dim=0,
            dim_size=inputs.num_samples,
            reduce="max",
        )
        all_reductions = torch.cat([sum_reduce, mean_reduce, maxmin_reduce], dim=-1)
        return self._map_to_size(all_reductions)


class HyperGNN(ModuleWithMetrics):
    def __init__(
        self,
        num_layers: int,
        hidden_state_size: int,
        node_emb_module: nn.Module,
        hyperedge_modules: List[nn.Module],
        hyperedge_type_module: nn.Module,
        hyperedge_arg_module: nn.Module,
        message_reduce_kind: Literal[
            "sum", "mean", "max", "min", "self-attention-max", "allset", "allsettransformer", "transformer", "pna-like"
        ] = "max",
        edge_update_type: Literal[None, "identity", "transformer"] = "transformer",
        node_update_type: Literal[
            None, "none", "linear", "linear-residual", "add-and-norm", "transformer"
        ] = "transformer",
        edge_dropout_rate: float = 0.0,
        dropout_rate: float = 0.0,
        normalise_by_node_degree: bool = False,
        transformer_feedforward_size: int = 2048,
    ):
        super().__init__()
        self.__hyperedge_type_module = hyperedge_type_module
        self.__hyperedge_arg_module = hyperedge_arg_module
        self.__node_emb_module = node_emb_module
        self.__hyperedge_modules = hyperedge_modules
        assert 0 <= edge_dropout_rate < 1
        assert 0 <= dropout_rate < 1
        self.__edge_dropout_rate = edge_dropout_rate
        self.__dropout_rate = dropout_rate
        self.__normalise_by_node_degree = normalise_by_node_degree

        self._reduce_kind = message_reduce_kind
        if message_reduce_kind in {"sum", "mean", "min", "max"}:
            reductor = SimpleVarSizedElementReduce(message_reduce_kind)
            self.__node_reductor = nn.ModuleList([reductor for _ in range(num_layers)])
        elif message_reduce_kind == "pna-like":
            reductor = PnaLikeVarSizedElementReduce(hidden_state_size)
            self.__node_reductor = nn.ModuleList([reductor for _ in range(num_layers)])
        elif message_reduce_kind == "self-attention-max":
            assert (
                not normalise_by_node_degree
            ), "Incompatible setting, node degree normalization doesn't make sense here."
            self.__node_reductor = nn.ModuleList(
                [
                    SelfAttentionVarSizedElementReduce(
                        hidden_state_size, hidden_state_size, hidden_state_size, SimpleVarSizedElementReduce("max")
                    )
                    for _ in range(num_layers)
                ]
            )
        elif message_reduce_kind == "allset":
            assert (
                not normalise_by_node_degree
            ), "Incompatible setting, node degree normalization doesn't make sense here."
            self.__node_reductor = nn.ModuleList(
                [
                    AllDeepSetElementReduce(hidden_state_size, "max", dropout_rate=dropout_rate)
                    for _ in range(num_layers)
                ]
            )
        elif message_reduce_kind == "allsettransformer":
            assert (
                not normalise_by_node_degree
            ), "Incompatible setting, node degree normalization doesn't make sense here."
            self.__node_reductor = nn.ModuleList(
                [
                    AllSetTransformerReduce(
                        hidden_state_size,
                        num_heads=8,
                        use_value_layer=True,
                        dropout_rate=dropout_rate,
                    )
                    for _ in range(num_layers)
                ]
            )
        elif message_reduce_kind == "transformer":
            assert (
                not normalise_by_node_degree
            ), "Incompatible setting, node degree normalization doesn't make sense here."
            self.__node_reductor = nn.ModuleList(
                [
                    MultiheadSelfAttentionVarSizedKeysElementReduce(
                        hidden_state_size,
                        hidden_state_size,
                        hidden_state_size,
                        num_heads=8,
                        dropout_rate=dropout_rate,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            raise ValueError("Unknown reduce kind `%s`" % message_reduce_kind)

        if node_update_type is None or node_update_type == "none":
            self._node_update_networks = nn.ModuleList([NoNodeUpdateNetwork() for _ in range(num_layers)])
        elif node_update_type == "linear":
            self._node_update_networks = nn.ModuleList(
                [LinearUpdateNetwork(hidden_state_size, dropout_rate, False) for _ in range(num_layers)]
            )
        elif node_update_type == "linear-residual":
            self._node_update_networks = nn.ModuleList(
                [LinearUpdateNetwork(hidden_state_size, dropout_rate, True) for _ in range(num_layers)]
            )
        elif node_update_type == "add-and-norm":
            self._node_update_networks = nn.ModuleList(
                [AddAndNorm(hidden_state_size, dropout_rate) for _ in range(num_layers)]
            )
        elif node_update_type == "transformer":
            self._node_update_networks = nn.ModuleList(
                [
                    TransformerUpdateNetwork(
                        hidden_state_size=hidden_state_size,
                        dropout_rate=dropout_rate,
                        feedfoward_dim=transformer_feedforward_size,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            raise ValueError(f"Unrecognized node update type `{node_update_type}`.")

        self._edge_update_type = edge_update_type

    @property
    def input_node_state_dim(self) -> int:
        return self.__node_emb_module.input_state_dimension

    @property
    def output_node_state_dim(self) -> int:
        return self.__hyperedge_modules[-1].output_state_dimension

    def _reset_module_metrics(self) -> None:
        self.__num_hyperedges = 0
        self.__num_hyperedges_args = 0

    def _module_metrics(self) -> Dict[str, Any]:
        return {
            "num_hyperedges": int(self.__num_hyperedges),
            "num_hyperedges_args": int(self.__num_hyperedges_args),
        }

    def forward(
        self,
        *,
        node_data,
        node_emb_idxs,
        hyperedge_data,
        hyperedge_unq_type,
        hyperedge_unq_arg_names,
        node_to_graph_idx,
        reference_node_graph_idx,
        reference_node_ids,
        num_graphs,
        num_edges,
        return_all_states: bool = False,
    ) -> GnnOutput:
        assert not return_all_states, "Not supported yet."
        initial_representations = self.__node_emb_module(**node_data)[node_emb_idxs]
        unq_hyperedge_type_reprs = self.__hyperedge_type_module(**hyperedge_unq_type)
        if self._edge_update_type is not None and self._edge_update_type != "none":
            edge_representations_state = unq_hyperedge_type_reprs[hyperedge_data["hyperedge_type_name_unq_idxs"]]
        else:
            edge_representations_state = None
        unq_hyperedge_arg_name_reprs = self.__hyperedge_arg_module(**hyperedge_unq_arg_names)
        nodes_state = initial_representations
        hyperedge_arg_node_idxs = hyperedge_data["hyperedge_arg_node_idxs"]
        if self.__normalise_by_node_degree:
            inverse_sqrt_node_degree = (
                torch_scatter.scatter(
                    torch.ones(hyperedge_arg_node_idxs.shape[0], device=hyperedge_arg_node_idxs.device),
                    index=hyperedge_arg_node_idxs,
                    dim_size=nodes_state.shape[0],
                    reduce="sum",
                )
                ** -0.5
            )
        else:
            inverse_sqrt_node_degree = None

        for i, hm in enumerate(self.__hyperedge_modules):
            messages_to_nodes, updated_edge_reprs = hm(
                unq_hyperedge_type_reprs=unq_hyperedge_type_reprs,
                unq_hyperedge_arg_name_reprs=unq_hyperedge_arg_name_reprs,
                nodes_representations=nodes_state,
                hyperedge_state_reprs=edge_representations_state,
                **hyperedge_data,
            )  # [num_args, hidden_size]

            if self._edge_update_type == "transformer":
                edge_representations_state = self._node_update_networks[i](
                    inputs=updated_edge_reprs, prev_state=edge_representations_state
                )
            elif self._edge_update_type == "identity":
                edge_representations_state = updated_edge_reprs

            if self.__edge_dropout_rate > 0 and self.training:
                edge_mask = torch.rand(num_edges, device=hyperedge_arg_node_idxs.device) < self.__edge_dropout_rate
                arg_mask = edge_mask[hyperedge_data["hyperedge_arg_to_edge_id"]]
                messages_to_nodes = messages_to_nodes[arg_mask]
                hyperedge_arg_node_idxs = hyperedge_arg_node_idxs[arg_mask]

            messages_to_nodes = ElementsToSummaryRepresentationInput(
                element_embeddings=messages_to_nodes,
                element_to_sample_map=hyperedge_arg_node_idxs,
                num_samples=initial_representations.shape[0],
            )

            if self._reduce_kind == "transformer":
                node_aggregated = self.__node_reductor[i](
                    messages_to_nodes,
                    queries=nodes_state,
                )  # [num_nodes, hidden_size]
            else:
                node_aggregated = self.__node_reductor[i](
                    messages_to_nodes,
                )  # [num_nodes, hidden_size]

            nodes_state = self._node_update_networks[i](inputs=node_aggregated, prev_state=nodes_state)

            if inverse_sqrt_node_degree is not None:
                nodes_state = nodes_state * inverse_sqrt_node_degree.unsqueeze(-1)

        with torch.no_grad():
            self.__num_hyperedges += num_edges
            self.__num_hyperedges_args += len(hyperedge_data["hyperedge_arg_name_unq_idxs"])
        return GnnOutput(
            input_node_representations=initial_representations,
            output_node_representations=nodes_state,
            node_to_graph_idx=node_to_graph_idx,
            node_idx_references=reference_node_ids,
            node_graph_idx_reference=reference_node_graph_idx,
            num_graphs=num_graphs,
        )


class HyperGnnModel(
    AbstractNeuralModel[
        HypergraphData,
        TensorizedHypergraphData,
        HyperGNN,
    ]
):

    LOGGER = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        node_representation_model: AbstractNeuralModel[TNodeData, TTensorizedNodeData, nn.Module],
        hyperedge_arg_model: AbstractNeuralModel[str, torch.Tensor, nn.Module],
        hyperedge_type_model: AbstractNeuralModel[str, torch.Tensor, nn.Module],
        max_nodes_per_graph: int = 80000,
        max_graph_hyperedges: int = 100000,
        max_memory: int = 1_000_000,
        max_nodes_per_hyperedge: int = 256,
        hidden_state_size: int = 128,
        stop_extending_minibatch_after_num_nodes: int = 10000,
        node_update_type: str = "transformer",
        edge_update_type: str = "transformer",
        hyperedge_representation_model: Optional[
            AbstractNeuralModel[HyperedgeData, TensorizedHyperedgeData, nn.Module]
        ] = None,
        reduce_kind: Literal["sum", "mean", "max", "min", "allset", "allsettransformer", "transformer"] = "max",
        edge_dropout_rate: int = 0.0,
        dropout_rate: int = 0.1,
        normalise_by_node_degree: bool = False,
    ):
        super().__init__()
        self._node_embedding_model: Final = node_representation_model
        self._hyperedge_arg_model: Final = hyperedge_arg_model
        self._hyperedge_type_model: Final = hyperedge_type_model
        self._hyperedge_model: Final = hyperedge_representation_model
        self.max_nodes_per_graph: Final = max_nodes_per_graph
        self.max_graph_hyperedges: Final = max_graph_hyperedges
        self.max_nodes_per_hyperedge: Final = max_nodes_per_hyperedge
        self.max_memory: Final = max_memory
        self.stop_extending_minibatch_after_num_nodes: Final = stop_extending_minibatch_after_num_nodes
        self._hidden_state_size = hidden_state_size
        self.__dropout_rate = dropout_rate
        self.__edge_dropout_rate = edge_dropout_rate
        self._node_update_type = node_update_type
        self._edge_update_type = edge_update_type

        assert reduce_kind in {
            "sum",
            "mean",
            "max",
            "min",
            "allset",
            "allsettransformer",
            "self-attention-max",
            "transformer",
            "pna-like",
        }
        self.__reduce_kind = reduce_kind
        self.__normalise_by_node_degree = normalise_by_node_degree

    def update_metadata_from(self, datapoint: HypergraphData):
        for node in datapoint.node_information:
            self._node_embedding_model.update_metadata_from(node)

        for h_edge in datapoint.hyperedges:
            for argname in h_edge.args:
                self._hyperedge_arg_model.update_metadata_from(argname)
            self._hyperedge_type_model.update_metadata_from(h_edge.hyperedge_type)
            self._hyperedge_model.update_metadata_from(h_edge)

        # TODO Don't ignore other metadata

    def tensorize(self, datapoint: HypergraphData) -> Optional[TensorizedHypergraphData]:
        if len(datapoint.node_information) > self.max_nodes_per_graph:
            self.LOGGER.warning("Dropping graph with %s nodes." % len(datapoint.node_information))
            return None
        if len(datapoint.hyperedges) > self.max_graph_hyperedges:
            self.LOGGER.warning("Dropping graph with %s hyperedges." % len(datapoint.hyperedge_types))
            return None

        # To reduce the cost of embeddings, we store every string only once, assign a unique ID to it,
        # and use that ID in all places where the string would appear:
        unq_nodes, unq_arg_names, unq_types = {}, {}, {}

        node_tensorized_data_idxs = np.fromiter(
            (_get_unique_idx(node_info, unq_nodes) for node_info in datapoint.node_information), dtype=np.int64
        )

        hyperedges = []
        memory_bound = 0
        num_hyperedges_removed = 0
        total_size_removed = 0
        for h_edge in datapoint.hyperedges:  # TODO: Should we shuffle to have a random order?
            if memory_bound + (len(h_edge.args) + 1) ** 2 > self.max_memory:
                num_hyperedges_removed += 1
                total_size_removed += (len(h_edge.args) + 1) ** 2
                continue
            else:
                memory_bound += (len(h_edge.args) + 1) ** 2
            # TODO: It would be nice if this would live in self._hyperedge_model.tensorize,
            # as that model's batching logic consumes this
            if len(h_edge.node_idxs) > self.max_nodes_per_hyperedge:
                LOGGER.warning(
                    f"Removing extra arguments from hyperedge of type `{h_edge.hyperedge_type}` with {len(h_edge.node_idxs)} arguments. Reducing to {self.max_nodes_per_hyperedge} args."
                )

            hyperedges.append(
                TensorizedHyperedgeData(
                    arg_name_unq_idxs=np.fromiter(
                        (_get_unique_idx(arg, unq_arg_names) for arg in h_edge.args[: self.max_nodes_per_hyperedge]),
                        dtype=np.int64,
                    ),
                    type_name_unq_idx=_get_unique_idx(h_edge.hyperedge_type, unq_types),
                    node_idxs=np.array(h_edge.node_idxs[: self.max_nodes_per_hyperedge], dtype=np.int64),
                    metadata=None,  # NOTE currently dropped
                )
            )
        if num_hyperedges_removed > 0:
            LOGGER.warning(
                f"Removed {num_hyperedges_removed} hyperedges because sample exceeded memory bound. Total removed size {total_size_removed}."
            )

        node_tensorized_unq_data = [self._node_embedding_model.tensorize(unq_node) for unq_node in unq_nodes]
        hyperedge_unq_types = [self._hyperedge_type_model.tensorize(unq_type) for unq_type in unq_types]
        hyperedge_unq_args = [self._hyperedge_arg_model.tensorize(unq_arg) for unq_arg in unq_arg_names]

        refno = {key: np.array(val) for key, val in datapoint.reference_nodes.items()}

        return TensorizedHypergraphData(
            node_tensorized_unq_data=node_tensorized_unq_data,  # List of unique node strings
            hyperedge_unq_types=hyperedge_unq_types,  # List of unique type names
            hyperedge_unq_args=hyperedge_unq_args,  # List of unique arg names
            node_tensorized_data_idxs=node_tensorized_data_idxs,  # List of node labels, as unique IDs to look up in node_tensorized_unq_data
            hyperedges=hyperedges,  # List of hyperedges, with arg/type names IDs to look up in hyperedge_unq_args/types
            reference_nodes=refno,  # reference nodes in the graphs
            num_nodes=len(datapoint.node_information),
        )

    def build_neural_module(self) -> HyperGNN:
        hyperedge_modules = self._hyperedge_model.build_neural_module()
        return HyperGNN(
            num_layers=len(hyperedge_modules),
            hidden_state_size=self._hidden_state_size,
            node_emb_module=self._node_embedding_model.build_neural_module(),
            hyperedge_modules=hyperedge_modules,
            hyperedge_type_module=self._hyperedge_type_model.build_neural_module(),
            hyperedge_arg_module=self._hyperedge_arg_model.build_neural_module(),
            message_reduce_kind=self.__reduce_kind,
            edge_update_type=self._edge_update_type,
            node_update_type=self._node_update_type,
            edge_dropout_rate=self.__edge_dropout_rate,
            dropout_rate=self.__dropout_rate,
            normalise_by_node_degree=self.__normalise_by_node_degree,
        )

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            "node_data_mb": self._node_embedding_model.initialize_minibatch(),
            "hyperedges": self._hyperedge_model.initialize_minibatch(),
            "hyperedge_unq_type": self._hyperedge_type_model.initialize_minibatch(),
            "hyperedge_unq_arg_names": self._hyperedge_arg_model.initialize_minibatch(),
            "node_emb_idxs": [],
            "unq_idx_cnt": {"types": 0, "args": 0, "nodes": 0},
            "hyperedges_node_idx": [],
            "num_nodes_per_graph": [],
            "reference_node_graph_idx": defaultdict(list),
            "reference_node_ids": defaultdict(list),
            "num_nodes_in_mb": 0,
            "num_edges": 0,
        }

    def extend_minibatch_with(
        self, tensorized_datapoint: TensorizedHypergraphData, partial_minibatch: Dict[str, Any]
    ) -> bool:
        continue_extending = True

        def extend_idxs(idxs, idxs_to_extend, num_idxs_so_far):
            return idxs.extend(idte + num_idxs_so_far for idte in idxs_to_extend)

        # Insert the node data
        for node_tensorized_data in tensorized_datapoint.node_tensorized_unq_data:
            continue_extending &= self._node_embedding_model.extend_minibatch_with(
                node_tensorized_data, partial_minibatch["node_data_mb"]
            )
        extend_idxs(
            partial_minibatch["node_emb_idxs"],
            tensorized_datapoint.node_tensorized_data_idxs,
            partial_minibatch["unq_idx_cnt"]["nodes"],
        )
        partial_minibatch["unq_idx_cnt"]["nodes"] += len(tensorized_datapoint.node_tensorized_unq_data)

        for h_type in tensorized_datapoint.hyperedge_unq_types:
            continue_extending &= self._hyperedge_type_model.extend_minibatch_with(
                h_type, partial_minibatch["hyperedge_unq_type"]
            )

        for arg in tensorized_datapoint.hyperedge_unq_args:
            continue_extending &= self._hyperedge_arg_model.extend_minibatch_with(
                arg, partial_minibatch["hyperedge_unq_arg_names"]
            )

        nodes_in_mb_so_far = partial_minibatch["num_nodes_in_mb"]

        type_idx_increase = partial_minibatch["unq_idx_cnt"]["types"]
        arg_idx_increase = partial_minibatch["unq_idx_cnt"]["args"]
        for hyperedge in tensorized_datapoint.hyperedges:
            partial_minibatch["num_edges"] += 1

            to_extend = TensorizedHyperedgeData(
                type_name_unq_idx=hyperedge.type_name_unq_idx + type_idx_increase,
                arg_name_unq_idxs=hyperedge.arg_name_unq_idxs + arg_idx_increase,
                node_idxs=hyperedge.node_idxs + nodes_in_mb_so_far,
                metadata=None,
            )
            continue_extending &= self._hyperedge_model.extend_minibatch_with(
                to_extend, partial_minibatch["hyperedges"]
            )

        partial_minibatch["unq_idx_cnt"]["types"] += len(tensorized_datapoint.hyperedge_unq_types)
        partial_minibatch["unq_idx_cnt"]["args"] += len(tensorized_datapoint.hyperedge_unq_args)

        graph_idx = len(partial_minibatch["num_nodes_per_graph"])

        for ref_name, ref_nodes in tensorized_datapoint.reference_nodes.items():
            partial_minibatch["reference_node_graph_idx"][ref_name].extend(graph_idx for _ in range(len(ref_nodes)))
            partial_minibatch["reference_node_ids"][ref_name].append(ref_nodes + nodes_in_mb_so_far)

        partial_minibatch["num_nodes_per_graph"].append(tensorized_datapoint.num_nodes)
        partial_minibatch["num_nodes_in_mb"] = nodes_in_mb_so_far + tensorized_datapoint.num_nodes
        return continue_extending & (
            partial_minibatch["num_nodes_in_mb"] < self.stop_extending_minibatch_after_num_nodes
        )

    @staticmethod
    def __create_node_to_graph_idx(num_nodes_per_graph: List[int]) -> Iterable[int]:
        for i, graph_size in enumerate(num_nodes_per_graph):
            yield from (i for _ in range(graph_size))

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        return {
            "hyperedge_unq_type": self._hyperedge_type_model.finalize_minibatch(
                accumulated_minibatch_data["hyperedge_unq_type"], device
            ),
            "hyperedge_unq_arg_names": self._hyperedge_arg_model.finalize_minibatch(
                accumulated_minibatch_data["hyperedge_unq_arg_names"], device
            ),
            "node_data": self._node_embedding_model.finalize_minibatch(
                accumulated_minibatch_data["node_data_mb"], device
            ),
            "node_emb_idxs": torch.tensor(accumulated_minibatch_data["node_emb_idxs"], device=device),
            "hyperedge_data": self._hyperedge_model.finalize_minibatch(
                accumulated_minibatch_data["hyperedges"], device
            ),
            "node_to_graph_idx": torch.tensor(
                np.fromiter(
                    self.__create_node_to_graph_idx(accumulated_minibatch_data["num_nodes_per_graph"]), dtype=np.int32
                ),
                dtype=torch.int64,
                device=device,
            ),
            "reference_node_graph_idx": {
                ref_name: torch.tensor(ref_node_graph_idx, dtype=torch.int64, device=device)
                for ref_name, ref_node_graph_idx in accumulated_minibatch_data["reference_node_graph_idx"].items()
            },
            "reference_node_ids": {
                ref_name: torch.tensor(np.concatenate(ref_node_idxs), dtype=torch.int64, device=device)
                for ref_name, ref_node_idxs in accumulated_minibatch_data["reference_node_ids"].items()
            },
            "num_graphs": len(accumulated_minibatch_data["num_nodes_per_graph"]),
            "num_edges": accumulated_minibatch_data["num_edges"],
        }
