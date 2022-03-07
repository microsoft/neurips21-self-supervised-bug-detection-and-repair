from typing import Any, Dict, List, Literal, TypeVar, Union

import logging
import torch
from ptgnn.baseneuralmodel import AbstractNeuralModel
from ptgnn.neuralmodels.reduceops.varsizedsummary import (
    ElementsToSummaryRepresentationInput,
    SimpleVarSizedElementReduce,
)
from torch import nn

from buglab.models.layers.allset_reduce import AllDeepSetElementReduce, AllSetTransformerReduce
from buglab.representations.data import TensorizedHyperedgeData

LOGGER = logging.getLogger(__name__)
TNodeData = TypeVar("TNodeData")
TNeuralModule = TypeVar("TNeuralModule")
TTensorizedNodeData = TypeVar("TTensorizedNodeData")
TensorizedHyperedgeArgs = TypeVar("TensorizedHyperedgeArgs")
HyperedgeData = TypeVar("HyperedgeData")


class HyperedgeMessagePassingModule(nn.Module):
    def __init__(
        self,
        hidden_state_size: int,
        argname_feature_state_size: int,
        reduce_kind: str = "max",
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.__hidden_state_size = hidden_state_size
        self.message_module = nn.Sequential(
            nn.Linear(hidden_state_size + argname_feature_state_size, hidden_state_size),
            nn.LeakyReLU(),
        )
        assert 0 <= dropout_rate < 1
        self.edge_update_module = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(2 * hidden_state_size + argname_feature_state_size, hidden_state_size),
            nn.LeakyReLU(),
        )
        if reduce_kind in {"sum", "mean", "min", "max"}:
            self.__reductor = SimpleVarSizedElementReduce(reduce_kind)
        elif reduce_kind == "allset":
            self.__reductor = AllDeepSetElementReduce(
                hidden_state_size + argname_feature_state_size, "sum", dropout_rate=dropout_rate
            )
        elif reduce_kind == "allsettransformer":
            self.__reductor = AllSetTransformerReduce(
                hidden_state_size + argname_feature_state_size,
                num_heads=8,
                use_value_layer=True,
                dropout_rate=dropout_rate,
            )

    @property
    def output_state_dimension(self):
        return self.__hidden_state_size

    def forward(
        self,
        *,
        nodes_representations,  # [num_nodes, hidden_size]
        hyperedge_arg_node_idxs,  # [num_args]
        unq_hyperedge_type_reprs,  # [|unq_types|, node_embedding + hidden_size]
        hyperedge_type_name_unq_idxs,  # [num_edges]
        unq_hyperedge_arg_name_reprs,  # [|unq_args|, arg_embedding]
        hyperedge_arg_name_unq_idxs,  # [num_args]
        hyperedge_arg_to_edge_id,  # [num_args]
        num_edges,
        hyperedge_state_reprs=None,  # if not None, shape is # [num_edges, arg_embedding + hidden_size]
        **kwargs,
    ):
        # Prepare tensor of (arg_name_repr || arg_node_repr):
        hyperedge_arg_name_reprs = unq_hyperedge_arg_name_reprs[
            hyperedge_arg_name_unq_idxs
        ]  # [num_args, arg_embedding]
        hyperedge_arg_node_reprs = nodes_representations[hyperedge_arg_node_idxs]  # [num_args, hidden_size]
        hyperedge_arg_reprs = torch.cat(
            [hyperedge_arg_name_reprs, hyperedge_arg_node_reprs], axis=-1
        )  # [num_args, arg_embedding + hidden_size]

        # Join with edge type representations:
        if hyperedge_state_reprs is None:
            hyperedge_state_reprs = unq_hyperedge_type_reprs[hyperedge_type_name_unq_idxs]  # [num_edges, hidden_size]
        if hyperedge_state_reprs.shape[1] == unq_hyperedge_type_reprs.shape[1]:
            # NOTE only pad with zeros if necessary
            hyperedge_state_reprs = torch.cat(
                (
                    hyperedge_state_reprs,
                    torch.zeros(
                        hyperedge_state_reprs.shape[0],
                        hyperedge_arg_name_reprs.shape[1],
                        device=hyperedge_state_reprs.device,
                    ),
                ),
                dim=-1,
            )  # [num_edges, arg_embedding + hidden_size]
        hyperedge_arg_and_type_reprs = torch.cat(
            [hyperedge_arg_reprs, hyperedge_state_reprs], axis=0
        )  # [num_args + num_edges, arg_embedding + hidden_size]

        # Create "messages" for all args / types:
        hyperedge_arg_and_type_msgs = self.message_module(
            hyperedge_arg_and_type_reprs
        )  # [num_args + num_edges, arg_embedding + hidden_size]

        # Aggregate these messages for each edge:
        hyperedge_arg_and_type_to_edge_idx = torch.cat(
            [hyperedge_arg_to_edge_id, torch.arange(num_edges, device=hyperedge_arg_to_edge_id.device)]
        )  # [num_args + num_edges, arg_embedding + hidden_size]

        edge_states = self.__reductor(
            ElementsToSummaryRepresentationInput(
                element_embeddings=hyperedge_arg_and_type_msgs,
                element_to_sample_map=hyperedge_arg_and_type_to_edge_idx,
                num_samples=num_edges,
            )
        )  # [num_edges, arg_embedding + hidden_size]
        message_to_nodes = self.edge_update_module(
            torch.cat(
                (
                    edge_states[hyperedge_arg_to_edge_id],  # [num_args, arg_embedding + hidden_size]
                    hyperedge_arg_reprs,  # [num_args, arg_embedding]
                ),
                dim=1,
            )  # [num_args, 2*arg_embedding + 2*hidden_size]
        )  # [num_args, hidden_size]

        return message_to_nodes, edge_states


class HyperedgeMessagePassingModel(
    AbstractNeuralModel[
        HyperedgeData,
        TensorizedHyperedgeData,
        List[HyperedgeMessagePassingModule],
    ]
):
    def __init__(
        self,
        *,
        message_input_state_size: int,
        hidden_state_size: int,
        argname_feature_state_size: int,
        dropout_rate: float = 0.1,
        num_layers: int = 8,
        tie_weights: bool = False,
        reduce_kind: Literal["sum", "mean", "max", "min", "allset", "allsettransformer"] = "max",
    ):
        super().__init__()
        assert 0 <= dropout_rate < 1
        assert reduce_kind in {"sum", "mean", "max", "min", "allset", "allsettransformer"}
        self.__dropout_rate = dropout_rate
        self.__message_input_state_size = message_input_state_size
        self.__hidden_state_size = hidden_state_size
        self.__argname_feature_state_size = argname_feature_state_size
        self.__num_layers = num_layers
        self.__tie_weights = tie_weights
        self.__reduce_kind = reduce_kind

    def tensorize(*args, **kwargs):
        pass

    def update_metadata_from(*args, **kwargs):
        pass

    def build_neural_module(self) -> List[HyperedgeMessagePassingModule]:
        if self.__tie_weights:
            module = HyperedgeMessagePassingModule(
                self.__hidden_state_size,
                self.__argname_feature_state_size,
                reduce_kind=self.__reduce_kind,
                dropout_rate=self.__dropout_rate,
            )
            module_list = [module for _ in range(self.__num_layers)]
        else:
            module_list = [
                HyperedgeMessagePassingModule(
                    self.__hidden_state_size,
                    self.__argname_feature_state_size,
                    reduce_kind=self.__reduce_kind,
                    dropout_rate=self.__dropout_rate,
                )
                for _ in range(self.__num_layers)
            ]
        return nn.ModuleList(module_list)

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            "hyperedge_arg_to_edge_id": [],
            "hyperedge_type_name_unq_idxs": [],
            "hyperedge_arg_name_unq_idxs": [],
            "hyperedge_arg_node_idxs": [],
            "num_edges_in_mb": 0,
        }

    def extend_minibatch_with(
        self, tensorized_datapoint: TensorizedHyperedgeData, partial_minibatch: Dict[str, Any]
    ) -> bool:
        partial_minibatch["hyperedge_arg_to_edge_id"].extend(
            partial_minibatch["num_edges_in_mb"] for _ in range(len(tensorized_datapoint.arg_name_unq_idxs))
        )
        partial_minibatch["hyperedge_type_name_unq_idxs"].append(tensorized_datapoint.type_name_unq_idx)
        partial_minibatch["hyperedge_arg_name_unq_idxs"].extend(tensorized_datapoint.arg_name_unq_idxs)
        partial_minibatch["hyperedge_arg_node_idxs"].extend(tensorized_datapoint.node_idxs)
        partial_minibatch["num_edges_in_mb"] += 1
        return True

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:

        return {
            "hyperedge_arg_to_edge_id": torch.tensor(
                accumulated_minibatch_data["hyperedge_arg_to_edge_id"], device=device
            ),
            "hyperedge_type_name_unq_idxs": torch.tensor(
                accumulated_minibatch_data["hyperedge_type_name_unq_idxs"], device=device
            ),
            "hyperedge_arg_name_unq_idxs": torch.tensor(
                accumulated_minibatch_data["hyperedge_arg_name_unq_idxs"], device=device
            ),
            "hyperedge_arg_node_idxs": torch.tensor(
                accumulated_minibatch_data["hyperedge_arg_node_idxs"], device=device
            ),
            "num_edges": accumulated_minibatch_data["num_edges_in_mb"],
        }
