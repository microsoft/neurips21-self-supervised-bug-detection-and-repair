from typing import Any, Dict, List, TypeVar, Union

import logging
import numpy as np
import torch
from collections import defaultdict
from numba import jit, prange
from numba.typed import List as nList
from ptgnn.baseneuralmodel import AbstractNeuralModel
from torch import nn
from torch.nn import MultiheadAttention

from buglab.representations.data import TensorizedHyperedgeData
from buglab.utils.bucketing import fast_greedy_bucket_elements

LOGGER = logging.getLogger(__name__)
HyperedgeData = TypeVar("HyperedgeData")


def calculate_seq_lens(max_size: int) -> List[int]:
    seq_lens = [16, 64, 256, 768, 1024]

    seq_lens = [b for b in seq_lens if b < max_size]
    next_multiple_of_8 = (max_size // 8 + int(max_size % 8 > 0)) * 8
    if len(seq_lens) > 0 and next_multiple_of_8 - seq_lens[-1] <= 32:
        seq_lens[-1] = next_multiple_of_8
    else:
        seq_lens.append(next_multiple_of_8)
    return seq_lens


def dummy_transformer(query, key, value, attn_mask):
    return (value,)


class HyperedgeTransformerModule(nn.Module):
    def __init__(
        self,
        message_input_state_size: int,
        dropout_rate: float,
        nheads: int = 8,
    ):
        super().__init__()
        assert 0 <= dropout_rate < 1

        self.__nheads = nheads
        self._dropout = nn.Dropout(dropout_rate)
        self.__arg_encoding_layer = nn.Linear(message_input_state_size, message_input_state_size)
        self.__hidden_state_size = message_input_state_size

        self.__message_module = MultiheadAttention(
            message_input_state_size,
            nheads,
            dropout=dropout_rate,
            batch_first=True,
        )

        # self.__message_module = dummy_transformer

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
        # maps seq_len to Long tensor of shape [|bucket_elements|, seq_len]:
        seq_len_to_arg_to_edge_id: Dict[int, torch.Tensor],
        # maps seq_len to Long tensor of shape [|bucket_elements|, seq_len]:
        seq_len_to_flattened_arg_idxs: Dict[int, torch.Tensor],
        hyperedge_state_reprs=None,
        **kwargs,
    ):
        hyperedge_arg_name_reprs = self.__arg_encoding_layer(self._dropout(unq_hyperedge_arg_name_reprs))[
            hyperedge_arg_name_unq_idxs
        ]  # [num_args, hidden_size]
        hyperedge_arg_node_reprs = nodes_representations[hyperedge_arg_node_idxs]  # [num_args, hidden_size]
        hyperedge_arg_reprs = hyperedge_arg_node_reprs + hyperedge_arg_name_reprs  # [num_args, hidden_size]

        # Join with edge type representations:
        if hyperedge_state_reprs is None:
            hyperedge_state_reprs = unq_hyperedge_type_reprs[hyperedge_type_name_unq_idxs]  # [num_edges, hidden_size]

        hyperedge_arg_and_type_reprs = torch.cat(
            (
                hyperedge_arg_reprs,
                hyperedge_state_reprs,
                # We create a dummy element that will be used as indexing target for the padding
                # values in our buckets (see finalize_minibatch for details of the indexing):
                torch.ones(
                    (1, hyperedge_state_reprs.shape[1]),
                    device=hyperedge_arg_reprs.device,
                    dtype=hyperedge_state_reprs.dtype,
                ),
            ),
            dim=0,
        )  # [num_args + num_edges + 1, hidden_size]

        messages_and_states = torch.empty(
            hyperedge_arg_node_idxs.shape[0] + hyperedge_state_reprs.shape[0] + 1,
            hyperedge_arg_and_type_reprs.shape[1],
            device=hyperedge_arg_and_type_reprs.device,
            dtype=nodes_representations.dtype,
        )

        # Start from longer buckets towards shorter ones. Starting with short ones which finish fast,
        #  cause wait time.
        for seq_len, bucketed_flattened_arg_idxs in reversed(seq_len_to_flattened_arg_idxs.items()):
            # Get the representations for all arguments appearing in all buckets of this size:
            bucketed_argnodes_and_type_embeddings = hyperedge_arg_and_type_reprs[
                bucketed_flattened_arg_idxs
            ]  # [|bucket_elements|, seq_len, arg_embedding + hidden_size]
            bucket_arg_to_edge_id = seq_len_to_arg_to_edge_id[seq_len]

            # Compute a mask such that elements can only attend to others from the same edge:
            mask = bucket_arg_to_edge_id.unsqueeze(-1) != bucket_arg_to_edge_id.unsqueeze(-2)
            mask = mask.unsqueeze(1).expand(-1, self.__nheads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])

            # Compute the actual messages using the Transformer layer:
            messages = self.__message_module(
                query=bucketed_argnodes_and_type_embeddings,
                key=bucketed_argnodes_and_type_embeddings,
                value=bucketed_argnodes_and_type_embeddings,
                attn_mask=mask,
            )[
                0
            ]  # [|bucket_elements|, seq_len, arg_embedding + hidden_size]

            messages = messages.reshape(-1, messages.shape[2])
            messages_and_states.scatter_(
                dim=0, index=bucketed_flattened_arg_idxs.view(-1).unsqueeze(-1).expand_as(messages), src=messages
            )

        messages_to_nodes, edge_states, _ = messages_and_states.split(
            [hyperedge_arg_node_idxs.shape[0], hyperedge_state_reprs.shape[0], 1]
        )

        # assert torch.allclose(messages_to_nodes, hyperedge_arg_reprs), (messages_to_nodes, hyperedge_arg_reprs)
        # assert torch.allclose(edge_states, hyperedge_state_reprs)

        return (
            messages_to_nodes,
            edge_states,
        )


@jit(nopython=False, forceobj=True, parallel=True)
def _numba_fill_bucket_sequences(
    index_buckets: nList[nList[int]],
    seq_len: int,
    num_hyperedges: int,
    num_hyperedge_args: int,
    hyperedge_sizes: np.ndarray,
    hyperedge_flattened_arg_idxs: List[int],
):
    """
    Takes a list of index buckets and uses them to fill two numpy tensors of shape [num_buckets, seq_len],
    where each index bucket is turned into a row (~ sequence) of the outputs.
    """
    # Pre-create arrays that we'll fill below; the fill values
    # will act as padding. For the edge_id, we can use -1; for the arg_idxs, we
    # use an index one higher than all real ones (which go up to (num HE args +
    # num HE types == num HE) - 1 because we do 0-based counting) and the model
    # will create a dummy value for this.
    num_buckets = len(index_buckets)
    shape = (num_buckets, seq_len)

    output_arg_to_edge_id = torch.full(shape, fill_value=-1, dtype=torch.int64)

    fill_value = num_hyperedges + num_hyperedge_args
    output_bucket_hyperedge_flattened_arg_idxs = torch.full(shape, fill_value=fill_value, dtype=torch.int64)

    for bucket_idx in prange(len(index_buckets)):
        start = 0
        bucket = index_buckets[bucket_idx]
        for hyperedge_idx in bucket:
            hyperedge_size = hyperedge_sizes[hyperedge_idx]
            output_arg_to_edge_id[bucket_idx, start : start + hyperedge_size] = hyperedge_idx
            # Add indices for the hyperedge arguments and type (together their number adds up to hyperedge_size):
            output_bucket_hyperedge_flattened_arg_idxs[
                bucket_idx, start : start + hyperedge_size - 1
            ] = torch.from_numpy(hyperedge_flattened_arg_idxs[hyperedge_idx])
            output_bucket_hyperedge_flattened_arg_idxs[bucket_idx, start + hyperedge_size - 1] = (
                hyperedge_idx + num_hyperedge_args
            )
            start += hyperedge_size

    return output_arg_to_edge_id, output_bucket_hyperedge_flattened_arg_idxs


class HyperedgeTransformerModel(
    AbstractNeuralModel[
        HyperedgeData,
        TensorizedHyperedgeData,
        HyperedgeTransformerModule,
    ]
):
    def __init__(
        self,
        *,
        hidden_state_size: int,
        argname_feature_state_size: int,
        dim_feedforward_transformer: int = 1024,
        dropout_rate: float = 0.0,
        num_layers: int = 8,
        tie_weights: bool = False,
        nheads: int = 8,
        norm_first: bool = False,
    ):
        super().__init__()
        self.__dim_feedforward_transformer = dim_feedforward_transformer
        self.__dropout_rate = dropout_rate
        self.__hidden_state_size = hidden_state_size
        self.__argname_feature_state_size = argname_feature_state_size
        self.__num_layers = num_layers
        self.__tie_weights = tie_weights
        self.__nheads = nheads
        self.__norm_first = norm_first

    def tensorize(*args, **kwargs):
        pass

    def update_metadata_from(*args, **kwargs):
        pass

    def build_neural_module(self) -> List[HyperedgeTransformerModule]:
        if self.__tie_weights:
            module = HyperedgeTransformerModule(
                self.__hidden_state_size,
                self.__dropout_rate,
                nheads=self.__nheads,
            )
            module_list = [module for _ in range(self.__num_layers)]
        else:
            module_list = [
                HyperedgeTransformerModule(
                    self.__hidden_state_size,
                    self.__dropout_rate,
                    nheads=self.__nheads,
                )
                for _ in range(self.__num_layers)
            ]
        return nn.ModuleList(module_list)

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            "hyperedges": [],
            "hyperedge_sizes": [],
            "hyperedge_type_name_unq_idxs": [],
            "hyperedge_arg_name_unq_idxs": [],
            "hyperedge_arg_node_idxs": [],
            "hyperedge_arg_to_edge_id": [],
            "hyperedge_flattened_arg_idxs": [],
            "num_hyperedges": 0,
            "num_hyperedge_args": 0,
        }

    def extend_minibatch_with(
        self, tensorized_datapoint: TensorizedHyperedgeData, partial_minibatch: Dict[str, Any]
    ) -> bool:
        num_hyperedge_args_in_graph = len(tensorized_datapoint.node_idxs)

        # Add the actual hyperedge information - note that these indices have already been
        # adjusted for being in a batch in the HyperGnnModel class
        partial_minibatch["hyperedge_type_name_unq_idxs"].append(tensorized_datapoint.type_name_unq_idx)
        partial_minibatch["hyperedge_arg_name_unq_idxs"].extend(tensorized_datapoint.arg_name_unq_idxs)
        partial_minibatch["hyperedge_arg_node_idxs"].extend(tensorized_datapoint.node_idxs)

        # Create a mapping from each added hyperedge argument node to the corresponding hyperedge
        num_hyperedges = partial_minibatch["num_hyperedges"]
        partial_minibatch["hyperedge_arg_to_edge_id"].extend(num_hyperedges for _ in range(num_hyperedge_args_in_graph))
        partial_minibatch["num_hyperedges"] += 1
        # Record which of the entries in partial_minibatch["hyperedge_arg_name_unq_idxs"] and
        # partial_minibatch["hyperedge_arg_node_idxs"] correspond to this hyperedge edge

        num_hyperedge_args = partial_minibatch["num_hyperedge_args"]
        partial_minibatch["hyperedge_flattened_arg_idxs"].append(
            np.arange(start=num_hyperedge_args, stop=num_hyperedge_args + num_hyperedge_args_in_graph, dtype=np.int32)
        )
        partial_minibatch["num_hyperedge_args"] += num_hyperedge_args_in_graph

        # Record the "size" of this hyperedge, which we'll use for our bucketing:
        partial_minibatch["hyperedge_sizes"].append(
            len(tensorized_datapoint.arg_name_unq_idxs) + 1
        )  # NOTE +1 for the type of the hyperedge

        # Record the original hyperedge for debugging/devel purposes
        partial_minibatch["hyperedges"].append(tensorized_datapoint)

        return True

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        hyperedge_sizes = np.array(accumulated_minibatch_data["hyperedge_sizes"])
        max_size = hyperedge_sizes.max()
        allowed_seq_lens = calculate_seq_lens(max_size)
        seq_lens, seq_len_to_indices = fast_greedy_bucket_elements(hyperedge_sizes, tuple(allowed_seq_lens))

        seq_len_to_index_buckets: Dict[int, List[List[int]]] = defaultdict(list)
        for seq_len, index_bucket in zip(seq_lens, seq_len_to_indices):
            seq_len_to_index_buckets[seq_len].append(index_bucket)

        numba_hyperedge_flattened_arg_idxs = accumulated_minibatch_data["hyperedge_flattened_arg_idxs"]
        num_hyperedge_args = accumulated_minibatch_data["num_hyperedge_args"]
        num_hyperedges = accumulated_minibatch_data["num_hyperedges"]
        hyperedge_sizes = np.array(accumulated_minibatch_data["hyperedge_sizes"])

        # For each seq_len, transform each corresponding bucket of indices to a single sequence.
        # (Core functionality offloaded to numba for speed reasons)
        seq_len_to_arg_to_edge_id, seq_len_to_flattened_arg_idxs = {}, {}
        for seq_len, index_buckets in seq_len_to_index_buckets.items():
            seq_len_arg_to_edge_id, seq_len_hyperedge_flattened_arg_idxs = _numba_fill_bucket_sequences(
                index_buckets,
                seq_len,
                num_hyperedges,
                num_hyperedge_args,
                hyperedge_sizes,
                numba_hyperedge_flattened_arg_idxs,
            )

            seq_len_to_arg_to_edge_id[seq_len] = seq_len_arg_to_edge_id.to(device)
            seq_len_to_flattened_arg_idxs[seq_len] = seq_len_hyperedge_flattened_arg_idxs.to(device)

        return {
            "hyperedge_type_name_unq_idxs": torch.tensor(
                accumulated_minibatch_data["hyperedge_type_name_unq_idxs"], device=device
            ),
            "hyperedge_arg_name_unq_idxs": torch.tensor(
                accumulated_minibatch_data["hyperedge_arg_name_unq_idxs"], device=device
            ),
            "hyperedge_arg_node_idxs": torch.tensor(
                accumulated_minibatch_data["hyperedge_arg_node_idxs"], device=device
            ),
            "hyperedge_arg_to_edge_id": torch.tensor(
                accumulated_minibatch_data["hyperedge_arg_to_edge_id"], device=device
            ),
            "seq_len_to_arg_to_edge_id": seq_len_to_arg_to_edge_id,
            "seq_len_to_flattened_arg_idxs": seq_len_to_flattened_arg_idxs,
        }
