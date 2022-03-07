from typing import List

import torch
from ptgnn.neuralmodels.reduceops.varsizedsummary import SimpleVarSizedElementReduce
from torch import nn
from torch_scatter import scatter_sum

from buglab.models.layers.hyperedge_message_passing import HyperedgeMessagePassingModel


class HGNN(nn.Module):
    """
    Implements the hypergraph neural network described in https://arxiv.org/abs/1809.09401

    The hyperedge convolution is based on eqn (10), assuming W=I (i.e. our edge weights are 1)
    """

    def __init__(
        self,
        hidden_state_size: int,
        argname_feature_state_size: int,
        dropout_rate: float = 0.0,
        use_arg_names: bool = False,
    ):
        super().__init__()
        self.__hidden_state_size = hidden_state_size
        self.linear_map = nn.Linear(hidden_state_size, hidden_state_size, bias=False)
        assert 0 <= dropout_rate < 1
        self.__reductor = SimpleVarSizedElementReduce("sum")
        self.__use_arg_names = use_arg_names
        if use_arg_names:
            self.__edge_dimensionality_remap = nn.Linear(
                hidden_state_size + argname_feature_state_size, hidden_state_size
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
        **kwargs,
    ):
        inverse_edge_degree = (
            scatter_sum(
                torch.ones(hyperedge_arg_node_idxs.shape[0], device=hyperedge_arg_node_idxs.device),
                index=hyperedge_arg_to_edge_id,
                dim_size=hyperedge_type_name_unq_idxs.shape[0],
            )
            ** -1
        )  # [num_edges]
        transformed_nodes_representations = self.linear_map(nodes_representations)
        hyperedge_arg_node_reprs = transformed_nodes_representations[hyperedge_arg_node_idxs]  # [num_args, hidden_size]

        if self.__use_arg_names:
            hyperedge_arg_name_reprs = unq_hyperedge_arg_name_reprs[
                hyperedge_arg_name_unq_idxs
            ]  # [num_args, arg_embedding]
            hyperedge_arg_reprs = torch.cat(
                [hyperedge_arg_name_reprs, hyperedge_arg_node_reprs], axis=-1
            )  # [num_args, arg_embedding + hidden_size]
        else:
            hyperedge_arg_reprs = hyperedge_arg_node_reprs  # [num_args, hidden_size]

        # Aggregate these representations for each edge:

        hyperedge_aggregated_msgs = scatter_sum(
            hyperedge_arg_reprs,
            index=hyperedge_arg_to_edge_id,
            dim=0,
            dim_size=num_edges,
        ) * inverse_edge_degree.unsqueeze(
            -1
        )  # [num_edges, hidden_size] or [num_edges, arg_embedding + hidden_size]

        if self.__use_arg_names:
            hyperedge_aggregated_msgs = self.__edge_dimensionality_remap(
                hyperedge_aggregated_msgs
            )  # [num_edges, hidden_size]

        return hyperedge_aggregated_msgs[hyperedge_arg_to_edge_id], hyperedge_aggregated_msgs


class HGNNModel(HyperedgeMessagePassingModel):
    def __init__(
        self,
        *,
        hidden_state_size: int,
        argname_feature_state_size: int,
        message_input_state_size: int,
        dropout_rate: float = 0.1,
        num_layers: int = 8,
        tie_weights: bool = False,
        use_arg_names: bool = False,
    ):
        super().__init__(
            message_input_state_size=message_input_state_size,
            hidden_state_size=hidden_state_size,
            argname_feature_state_size=argname_feature_state_size,
            dropout_rate=dropout_rate,
            num_layers=num_layers,
            tie_weights=tie_weights,
        )
        assert 0 <= dropout_rate < 1
        self.__dropout_rate = dropout_rate
        self.__hidden_state_size = hidden_state_size
        self.__argname_feature_state_size = argname_feature_state_size
        self.__num_layers = num_layers
        self.__tie_weights = tie_weights
        self.__use_arg_names = use_arg_names

    def build_neural_module(self) -> List[HGNN]:
        if self.__tie_weights:
            module = HGNN(
                self.__hidden_state_size,
                self.__argname_feature_state_size,
                dropout_rate=self.__dropout_rate,
                use_arg_names=self.__use_arg_names,
            )
            module_list = [module for _ in range(self.__num_layers)]
        else:
            module_list = [
                HGNN(
                    self.__hidden_state_size,
                    self.__argname_feature_state_size,
                    dropout_rate=self.__dropout_rate,
                    use_arg_names=self.__use_arg_names,
                )
                for _ in range(self.__num_layers)
            ]
        return nn.ModuleList(module_list)
