from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Tuple, Union

import logging
import numpy as np
import torch
from collections import defaultdict
from ptgnn.baseneuralmodel import AbstractNeuralModel
from ptgnn.neuralmodels.embeddings.strelementrepresentationmodel import (
    CharUnitEmbedder,
    StrElementRepresentationModel,
    SubtokenUnitEmbedder,
    TokenUnitEmbedder,
)
from torch import nn

from buglab.models.basemodel import AbstractBugLabModel, RewriteModuleSelectionInformation
from buglab.models.layers.relational_transformer import RelationalTransformerEncoderLayer
from buglab.models.rewrite_chooser_module import RewriteChooserInformation, RewriteChooserModule
from buglab.representations.data import BugLabData, BugLabGraph

LOGGER = logging.getLogger(__name__)


class SeqModelTensorizedSample(NamedTuple):
    input_token_ids: List[Union[int, List[int], np.ndarray]]
    intra_token_edges: Dict[str, List[Tuple[int, int]]]

    rewritable_loc_idxs: List[int]
    rewritten_node_idx: Optional[int]

    node_mappings: Dict[int, int]

    # Bug repair
    text_rewrites_info: RewriteModuleSelectionInformation
    varswap_rewrites_info: RewriteModuleSelectionInformation
    argswap_rewrites_info: RewriteModuleSelectionInformation

    num_rewrite_locations_considered: int

    # Bug selection
    observed_rewrite_logprobs: Optional[List[float]]


class SeqBugLabModule(RewriteChooserModule):
    def __init__(
        self,
        token_embedder: Union[TokenUnitEmbedder, SubtokenUnitEmbedder, CharUnitEmbedder],
        embedding_dim: int,
        num_edge_types: int,
        num_layers: int,
        num_heads: int,
        intermediate_dimension: int,
        dropout_rate: float,
        layer_type: Literal["great", "rat", "transformer", "gru"] = "great",
        rezero_mode: Literal["off", "scalar", "vector"] = "off",
        normalisation_mode: Literal["off", "prenorm", "postnorm"] = "postnorm",
        **kwargs,
    ):
        super().__init__(entity_repr_size=embedding_dim, **kwargs)
        self.__token_embedder = token_embedder

        self.__positional_encoding = nn.Parameter(torch.randn(1, 5000, embedding_dim), requires_grad=True)

        self.__dropout_layer = nn.Dropout(dropout_rate)
        self.__input_layer_norm = nn.LayerNorm(embedding_dim)

        self.__layer_type = layer_type
        if layer_type in {"great", "rat"}:
            self.__seq_layers = nn.ModuleList(
                [
                    RelationalTransformerEncoderLayer(
                        nhead=num_heads,
                        num_edge_types=num_edge_types,
                        d_model=embedding_dim,
                        key_query_dimension=embedding_dim // num_heads,
                        value_dimension=embedding_dim // num_heads,
                        dim_feedforward=intermediate_dimension,
                        dropout=dropout_rate,
                        use_edge_value_biases=layer_type == "rat",
                        rezero_mode=rezero_mode,
                        normalisation_mode=normalisation_mode,
                    )
                    for _ in range(num_layers)
                ]
            )
        elif layer_type == "transformer":
            self.__seq_layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=embedding_dim,
                        nhead=num_heads,
                        dim_feedforward=intermediate_dimension,
                        dropout=dropout_rate,
                    )
                    for _ in range(num_layers)
                ]
            )
        elif layer_type == "gru":
            self.__seq_layers = nn.GRU(
                input_size=embedding_dim,
                hidden_size=embedding_dim // 2,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unrecognized layer type `{layer_type}`.")

    def _compute_code_representation(
        self, input_sequence_ids, input_seq_num_subtokens, token_sequence_lengths, edges, edge_types
    ):
        output_representation = self.__token_embedder(
            token_idxs=input_sequence_ids.reshape(-1, input_sequence_ids.shape[-1]),
            lengths=input_seq_num_subtokens.reshape(-1),
        )  # [B*max_seq_len, D]

        output_representation = output_representation.reshape(
            input_sequence_ids.shape[0], input_sequence_ids.shape[1], -1
        )  # [B, max_seq_len, D]
        token_range = torch.arange(input_sequence_ids.shape[1], device=token_sequence_lengths.device)
        token_mask = token_range.unsqueeze(0) < token_sequence_lengths.unsqueeze(-1)  # [B, max_seq_len]

        if self.__layer_type in {"rat", "transformer", "great"}:
            # Add positional embeddings only to token nodes (and not to symbol nodes)

            output_representation = output_representation + self.__positional_encoding[:, : input_sequence_ids.shape[1]]

            # The original BERT model has this layer after embedding + positional encoding
            output_representation = self.__dropout_layer(self.__input_layer_norm(output_representation))
            seq_mask = token_mask.logical_not()  # [B, max_seq_len]

        output_representation *= token_mask.unsqueeze(-1)

        if self.__layer_type in {"rat", "great"}:
            for layer in self.__seq_layers:
                output_representation = layer(
                    src=output_representation, src_mask=seq_mask, edges=edges, edge_types=edge_types
                )  # [B, max_seq_len, H]
        elif self.__layer_type == "transformer":
            for layer in self.__seq_layers:
                output_representation = output_representation.transpose(0, 1)  # [max_seq_len, B, D]
                output_representation = layer(output_representation, src_key_padding_mask=seq_mask)
                output_representation = output_representation.transpose(0, 1)  # [B, max_seq_len, D]
        elif self.__layer_type == "gru":
            output_representation, _ = self.__seq_layers(
                nn.utils.rnn.pack_padded_sequence(
                    output_representation, lengths=token_sequence_lengths, batch_first=True, enforce_sorted=False
                )
            )
            output_representation, lens = nn.utils.rnn.pad_packed_sequence(output_representation, batch_first=True)
        else:
            raise ValueError()

        return output_representation

    def compute_rewrite_chooser_information(
        self,
        *,
        # Model-specific representation of code:
        input_sequence_ids,
        input_seq_num_subtokens,
        token_sequence_lengths,
        edges,
        edge_types,
        # Rewrite info:
        rewritable_loc_idxs: torch.Tensor,
        # Text repair
        text_rewrite_tok_idxs: torch.Tensor,
        text_rewrite_replacement_ids: torch.Tensor,
        text_rewrite_to_loc_group: torch.Tensor,
        # Var Misuse
        varswap_tok_idxs: torch.Tensor,
        varswap_replacement_tok_idxs: torch.Tensor,
        varswap_to_loc_group: torch.Tensor,
        # ArgSwap
        argswap_tok_idxs: torch.Tensor,
        argswap_replacement_tok_idxs: torch.Tensor,
        argswap_to_loc_group: torch.Tensor,
        # Other inputs that we don't consume:
        **kwargs,
    ) -> RewriteChooserInformation:
        code_representation = self._compute_code_representation(
            input_sequence_ids, input_seq_num_subtokens, token_sequence_lengths, edges, edge_types
        )  # [B, max_seq_len, D]

        return RewriteChooserInformation(
            num_samples=code_representation.shape[0],
            rewritable_loc_reprs=code_representation[rewritable_loc_idxs[:, 0], rewritable_loc_idxs[:, 1]],
            rewritable_loc_to_sample_id=rewritable_loc_idxs[:, 0],
            text_rewrite_loc_reprs=code_representation[text_rewrite_tok_idxs[:, 0], text_rewrite_tok_idxs[:, 1]],
            text_rewrite_replacement_ids=text_rewrite_replacement_ids,
            text_rewrite_to_loc_group=text_rewrite_to_loc_group,
            varswap_loc_reprs=code_representation[varswap_tok_idxs[:, 0], varswap_tok_idxs[:, 1]],
            varswap_replacement_reprs=code_representation[
                varswap_replacement_tok_idxs[:, 0], varswap_replacement_tok_idxs[:, 1]
            ],
            varswap_to_loc_group=varswap_to_loc_group,
            argswap_loc_reprs=code_representation[argswap_tok_idxs[:, 0], argswap_tok_idxs[:, 1]],
            # This is special, because argswap_replacement_tok_idxs has shape [?, 3], with the first being the sample
            # index and the remaining ones being the token indices in that sample.
            argswap_replacement_reprs=code_representation[
                argswap_replacement_tok_idxs[:, 0].unsqueeze(-1), argswap_replacement_tok_idxs[:, 1:]
            ],
            argswap_to_loc_group=argswap_to_loc_group,
        )

    def forward(self, **kwargs):
        rc_info = self.compute_rewrite_chooser_information(**kwargs)
        return self.compute_loss(rc_info=rc_info, **kwargs)


class SeqBugLabModel(AbstractNeuralModel[BugLabData, SeqModelTensorizedSample, SeqBugLabModule], AbstractBugLabModel):
    def __init__(
        self,
        representation_size: int,
        max_subtoken_vocab_size: int,
        dropout_rate: float,
        layer_type: Literal["great", "rat", "transformer", "gru"] = "great",
        max_seq_size: int = 500,
        num_heads: int = 8,
        num_layers: int = 6,
        intermediate_dimension_size: int = 2048,
        rezero_mode: Literal["off", "scalar", "vector"] = "off",
        normalisation_mode: Literal["off", "prenorm", "postnorm"] = "postnorm",
        **kwargs,
    ):
        super().__init__()
        AbstractBugLabModel.__init__(self, **kwargs)
        self._init()
        self._edge_types = set()

        self._dropout_rate = dropout_rate
        self._layer_type = layer_type

        self._max_seq_size = max_seq_size
        self._token_embedder = StrElementRepresentationModel(
            token_splitting="subtoken",
            embedding_size=representation_size,
            dropout_rate=dropout_rate,
            vocabulary_size=max_subtoken_vocab_size,
            subtoken_combination="max",
        )

        self._num_heads = num_heads
        self._num_layers = num_layers
        self._intermediate_dimension_size = intermediate_dimension_size
        self._rezero_mode = rezero_mode
        self._normalisation_mode = normalisation_mode

    def __to_token_data(self, graph: BugLabGraph):
        # Get token list
        token_sequence, next_token_edges = self.__extract_token_sequence(graph)
        if len(token_sequence) == 0:
            return None
        old_node_id_to_new = {t: i for i, t in enumerate(token_sequence)}

        # Get all edges among tokens
        NON_TOKEN_EDGES = frozenset(
            {
                "NextToken",
                "PossibleType",
                "CandidateCall",
                "CandidateCallDoc",
                "MayFormalName",
                "Child",
                "Sibling",
                "OccurrenceOf",
            }
        )

        children = defaultdict(list)
        for child_edge in graph["edges"]["Child"]:
            f, t = child_edge[:2]
            children[f].append(t)

        def get_token_node(from_node_idx):
            to_visit = [from_node_idx]
            while len(to_visit) > 0:
                next_node = to_visit.pop()
                if next_node in token_sequence:
                    return next_node
                to_visit.extend(children[next_node])
            assert False, "No token found from the root node index."

        def backoff_to_parent(node_idx):
            return to_token_idx(next((k for k, v in children.items() if node_idx in v)))

        def to_token_idx(old_node_idx) -> int:
            if old_node_idx in old_node_id_to_new:
                return old_node_id_to_new[old_node_idx]
            to_visit = [old_node_idx]
            while len(to_visit) > 0:
                next_node_idx = to_visit.pop()
                next_node_label = graph["nodes"][next_node_idx]

                if next_node_label == "ComparisonTarget":
                    for child_node_idx in children[next_node_idx]:
                        child_node_label = graph["nodes"][child_node_idx]
                        if child_node_label in {
                            "<",
                            "<=",
                            "==",
                            "!=",
                            ">",
                            ">=",
                            "is",
                            "in",
                            "not",
                        }:
                            return old_node_id_to_new[child_node_idx]
                        elif child_node_label in {"IsNot", "NotIn"}:
                            # There are at least two tokens for these cases
                            return old_node_id_to_new[get_token_node(child_node_idx)]
                    else:
                        assert False, "One of the children, should be a comparison operator."
                elif next_node_label == "BinaryOperation":
                    for child_node_idx in children[next_node_idx]:
                        child_node_label = graph["nodes"][child_node_idx]
                        if child_node_label in {"+", "-", "*", "/", "//", "**", "%", "@", ">>", "<<", "|", "&", "^"}:
                            if child_node_idx not in old_node_id_to_new:
                                return backoff_to_parent(old_node_idx)
                            return old_node_id_to_new[child_node_idx]
                    else:
                        assert False, "One of the children, should be a binary operator."
                elif next_node_label in ("Assign", "AugAssign"):
                    for assign_target_node_idx in children[next_node_idx]:
                        if "=" in graph["nodes"][assign_target_node_idx]:
                            break
                    else:
                        assert False, "One child should always be an equals symbol."

                    return old_node_id_to_new[assign_target_node_idx]
                else:
                    for child_node_idx in children[next_node_idx]:
                        child_token_idx = old_node_id_to_new.get(child_node_idx)
                        if child_token_idx is not None:
                            return child_token_idx
                        else:
                            to_visit.append(child_node_idx)

            # Backoff to parent, should be rarely necessary (e.g. f-strings)
            return backoff_to_parent(old_node_idx)

        # Get ids of all AST nodes
        for child_edge in graph["edges"]["Child"]:
            f, t = child_edge[:2]
            old_node_id_to_new[f] = to_token_idx(f)
            old_node_id_to_new[t] = to_token_idx(t)

        # Map symbol nodes/edges to their first token occurrence
        orignal_symbol_node_idxs = defaultdict(list)
        for f, t in graph["edges"]["OccurrenceOf"]:
            orignal_symbol_node_idxs[t].append(f)
        for symbol_node_idx, target_tokens in orignal_symbol_node_idxs.items():
            old_node_id_to_new[symbol_node_idx] = min(old_node_id_to_new[t] for t in target_tokens)

        # Replace the edges
        new_edges = {}
        for edge_type, adj_list in graph["edges"].items():
            if edge_type in NON_TOKEN_EDGES:
                continue
            new_edges[edge_type] = [(to_token_idx(from_idx), to_token_idx(to_idx)) for from_idx, to_idx in adj_list]

        node_labels = graph["nodes"]
        all_tokens = [node_labels[i] for i in token_sequence]

        ref_token_idxs = []
        for ref_node_idx in graph["reference_nodes"]:
            ref_node_new_idx = to_token_idx(ref_node_idx)
            old_node_id_to_new[ref_node_idx] = ref_node_new_idx
            ref_token_idxs.append(ref_node_new_idx)

        return (
            all_tokens,
            old_node_id_to_new,
            new_edges,
            ref_token_idxs,
        )

    def __extract_token_sequence(self, graph: BugLabGraph) -> List[int]:
        next_token_edges = {f: t for f, t in graph["edges"]["NextToken"]}
        first_token = set(next_token_edges.keys()) - set(next_token_edges.values())
        if len(first_token) != 1:
            LOGGER.error(f"Encountered graph where the tokens are not connected in {graph['path']}")
            return []
        first_token_idx = next(iter(first_token))

        def get_token_sequence():
            seen_tokens = {first_token_idx}
            next_token = first_token_idx
            yield next_token
            while next_token in next_token_edges:
                next_token = next_token_edges[next_token]
                if next_token in seen_tokens:
                    LOGGER.error("Cyclic token sequence in %s", graph["path"])
                    raise Exception()
                seen_tokens.add(next_token)
                yield next_token

        try:
            token_sequence = list(get_token_sequence())
        except Exception:
            return []
        if len(token_sequence) != len(next_token_edges) + 1:
            LOGGER.error("Broken token sequence in %s", graph["path"])
            return []
        return token_sequence, next_token_edges

    def update_metadata_from(self, datapoint: BugLabData) -> None:
        try:
            token_data = self.__to_token_data(datapoint["graph"])
        except Exception as ex:
            LOGGER.exception("Error in generating token sequence for %s", datapoint["graph"]["path"], exc_info=ex)
            return None
        if token_data is None:
            return None
        node_labels, _, edges, _ = token_data

        for n in node_labels:
            self._token_embedder.update_metadata_from(n)

        self._edge_types.update(edges.keys())

    def finalize_metadata(self) -> None:
        self._edge_types = list(self._edge_types)
        self._edge_type_to_idx = {t: i for i, t in enumerate(self._edge_types)}

    def build_neural_module(self) -> SeqBugLabModule:
        return SeqBugLabModule(
            token_embedder=self._token_embedder.build_neural_module(),
            embedding_dim=self._token_embedder.embedding_size,
            num_edge_types=len(self._edge_types),
            num_layers=self._num_layers,
            num_heads=self._num_heads,
            intermediate_dimension=self._intermediate_dimension_size,
            dropout_rate=self._dropout_rate,
            rewrite_vocabulary_size=len(self._target_rewrite_ops),
            layer_type=self._layer_type,
            buggy_samples_weight_schedule=self._buggy_samples_weight_schedule,
            repair_weight_schedule=self._repair_weight_schedule,
            generator_loss_type=self._generator_loss_type,
            rezero_mode=self._rezero_mode,
            normalisation_mode=self._normalisation_mode,
            localization_module_type=self._localization_module_type,
        )

    def tensorize(self, datapoint: BugLabData) -> Optional[SeqModelTensorizedSample]:
        try:
            token_data = self.__to_token_data(datapoint["graph"])
        except Exception as ex:
            LOGGER.exception("Error in generating token sequence for %s", datapoint["graph"]["path"], exc_info=ex)
            return None
        if token_data is None:
            return None
        node_labels, old_node_id_to_new, edges, reference_nodes = token_data

        if len(node_labels) > self._max_seq_size:
            LOGGER.debug("Rejecting sample with %s tokens.", len(node_labels))
            return None

        # Note: the transform_candidate_node_idxs defined below may contain duplicates. This is thanks
        # to the non-unique mapping of nodes from the graph to the sequence.
        candidate_node_idxs, inv = np.unique(datapoint["graph"]["reference_nodes"], return_inverse=True)
        transformed_candidate_node_idxs = np.array([old_node_id_to_new[n] for n in candidate_node_idxs])

        if datapoint["target_fix_action_idx"] is not None:
            target_node_idx = inv[datapoint["target_fix_action_idx"]]
            assert (
                old_node_id_to_new[datapoint["graph"]["reference_nodes"][datapoint["target_fix_action_idx"]]]
                == transformed_candidate_node_idxs[target_node_idx]
            )
        else:
            target_node_idx = None

        (
            text_rewrite_selection_info,
            varswap_rewrite_selection_info,
            argswap_rewrite_selection_info,
            num_rewrite_locations,
        ) = self._compute_rewrite_data(datapoint, candidate_node_idxs)

        # Rewrite node indices according to our project:
        text_rewrite_selection_info = text_rewrite_selection_info._replace(
            node_idxs=[old_node_id_to_new[n] for n in text_rewrite_selection_info.node_idxs]
        )
        varswap_rewrite_selection_info = varswap_rewrite_selection_info._replace(
            node_idxs=[old_node_id_to_new[n] for n in varswap_rewrite_selection_info.node_idxs],
            replacement_ids=[old_node_id_to_new[n] for n in varswap_rewrite_selection_info.replacement_ids],
        )
        argswap_rewrite_selection_info = argswap_rewrite_selection_info._replace(
            node_idxs=[old_node_id_to_new[n] for n in argswap_rewrite_selection_info.node_idxs],
            replacement_ids=[
                (old_node_id_to_new[p1], old_node_id_to_new[p2])
                for p1, p2 in argswap_rewrite_selection_info.replacement_ids
            ],
        )

        return SeqModelTensorizedSample(
            input_token_ids=[self._token_embedder.tensorize(t) for t in node_labels],
            intra_token_edges=edges,
            rewritable_loc_idxs=transformed_candidate_node_idxs,
            rewritten_node_idx=target_node_idx,
            node_mappings=old_node_id_to_new,
            # Bug repair
            text_rewrites_info=text_rewrite_selection_info,
            varswap_rewrites_info=varswap_rewrite_selection_info,
            argswap_rewrites_info=argswap_rewrite_selection_info,
            num_rewrite_locations_considered=num_rewrite_locations,
            # Rewrite logprobs
            observed_rewrite_logprobs=datapoint.get("candidate_rewrite_logprobs", None),
        )

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            "input_subtoken_ids": [],
            "edges": [],
            "edge_types": [],
            "rewritable_loc_idxs": [],
            "sample_has_bug": [],
            "sample_to_correct_loc_idx": [],
            "node_mappings": [],
            # Repair
            "mb_num_repair_groups": 0,
            "mb_num_rewrite_candidates": 0,
            # Text rewrites
            "text_rewrite_tok_idxs": [],
            "text_rewrite_replacement_ids": [],
            "text_rewrite_to_loc_group": [],
            "text_rewrite_correct_idxs": [],
            "text_rewrite_original_idxs": [],
            "text_rewrite_joint_idxs": [],
            # Var Misuse
            "varswap_tok_idxs": [],
            "varswap_replacement_tok_idxs": [],
            "varswap_to_loc_group": [],
            "varswap_correct_idxs": [],
            "varswap_original_idxs": [],
            "varswap_joint_idxs": [],
            # Arg Swaps
            "argswap_tok_idxs": [],
            "argswap_replacement_tok_idxs": [],
            "argswap_to_loc_group": [],
            "argswap_correct_idxs": [],
            "argswap_original_idxs": [],
            "argswap_joint_idxs": [],
            # Rewrite logprobs
            "rewrite_to_sample_id": [],
            "observed_rewrite_logprobs": [],
            "no_bug_observed_rewrite_logprobs": [],
        }

    def extend_minibatch_with(
        self, tensorized_datapoint: SeqModelTensorizedSample, partial_minibatch: Dict[str, Any]
    ) -> bool:
        current_sample_idx = len(partial_minibatch["input_subtoken_ids"])

        partial_minibatch["input_subtoken_ids"].append(tensorized_datapoint.input_token_ids)
        for edge_type, adj_list in tensorized_datapoint.intra_token_edges.items():
            if edge_type not in self._edge_type_to_idx:
                continue
            partial_minibatch["edge_types"].extend(self._edge_type_to_idx[edge_type] for _ in adj_list)
            partial_minibatch["edges"].extend((current_sample_idx, f, t) for f, t in adj_list)
        partial_minibatch["node_mappings"].append(tensorized_datapoint.node_mappings)

        if tensorized_datapoint.rewritten_node_idx is None:
            partial_minibatch["sample_has_bug"].append(False)
            partial_minibatch["sample_to_correct_loc_idx"].append(-10000)  # Marker for "not buggy"
        else:
            partial_minibatch["sample_has_bug"].append(True)
            partial_minibatch["sample_to_correct_loc_idx"].append(
                tensorized_datapoint.rewritten_node_idx + len(partial_minibatch["rewritable_loc_idxs"])
            )

        partial_minibatch["rewritable_loc_idxs"].extend(
            (current_sample_idx, tid) for tid in tensorized_datapoint.rewritable_loc_idxs
        )

        mb_rewrite_candidate_offset = partial_minibatch["mb_num_rewrite_candidates"]
        num_rewrite_candidates_in_datapoint = 0
        mb_num_repair_groups_offset = partial_minibatch["mb_num_repair_groups"]

        # Text rewrites
        if tensorized_datapoint.text_rewrites_info.correct_choice_idx is not None:
            partial_minibatch["text_rewrite_correct_idxs"].append(
                tensorized_datapoint.text_rewrites_info.correct_choice_idx
                + len(partial_minibatch["text_rewrite_to_loc_group"])
            )
        partial_minibatch["text_rewrite_tok_idxs"].extend(
            (current_sample_idx, t) for t in tensorized_datapoint.text_rewrites_info.node_idxs
        )
        partial_minibatch["text_rewrite_replacement_ids"].extend(
            tensorized_datapoint.text_rewrites_info.replacement_ids
        )
        partial_minibatch["text_rewrite_to_loc_group"].extend(
            t + mb_num_repair_groups_offset for t in tensorized_datapoint.text_rewrites_info.loc_groups
        )
        partial_minibatch["text_rewrite_joint_idxs"].extend(
            t + mb_rewrite_candidate_offset for t in tensorized_datapoint.text_rewrites_info.original_idxs
        )
        num_rewrite_candidates_in_datapoint += len(tensorized_datapoint.text_rewrites_info.original_idxs)

        # Var Misuse
        if tensorized_datapoint.varswap_rewrites_info.correct_choice_idx is not None:
            partial_minibatch["varswap_correct_idxs"].append(
                tensorized_datapoint.varswap_rewrites_info.correct_choice_idx
                + len(partial_minibatch["varswap_to_loc_group"])
            )
        partial_minibatch["varswap_tok_idxs"].extend(
            (current_sample_idx, t) for t in tensorized_datapoint.varswap_rewrites_info.node_idxs
        )
        partial_minibatch["varswap_replacement_tok_idxs"].extend(
            (current_sample_idx, t) for t in tensorized_datapoint.varswap_rewrites_info.replacement_ids
        )
        partial_minibatch["varswap_to_loc_group"].extend(
            t + mb_num_repair_groups_offset for t in tensorized_datapoint.varswap_rewrites_info.loc_groups
        )
        partial_minibatch["varswap_joint_idxs"].extend(
            t + mb_rewrite_candidate_offset for t in tensorized_datapoint.varswap_rewrites_info.original_idxs
        )
        num_rewrite_candidates_in_datapoint += len(tensorized_datapoint.varswap_rewrites_info.original_idxs)

        # Arg Swaps
        if tensorized_datapoint.argswap_rewrites_info.correct_choice_idx is not None:
            partial_minibatch["argswap_correct_idxs"].append(
                tensorized_datapoint.argswap_rewrites_info.correct_choice_idx
                + len(partial_minibatch["argswap_to_loc_group"])
            )
        partial_minibatch["argswap_tok_idxs"].extend(
            (current_sample_idx, t) for t in tensorized_datapoint.argswap_rewrites_info.node_idxs
        )
        partial_minibatch["argswap_replacement_tok_idxs"].extend(
            (current_sample_idx, t1, t2) for t1, t2 in tensorized_datapoint.argswap_rewrites_info.replacement_ids
        )
        partial_minibatch["argswap_to_loc_group"].extend(
            t + mb_num_repair_groups_offset for t in tensorized_datapoint.argswap_rewrites_info.loc_groups
        )
        partial_minibatch["argswap_joint_idxs"].extend(
            t + mb_rewrite_candidate_offset for t in tensorized_datapoint.argswap_rewrites_info.original_idxs
        )
        num_rewrite_candidates_in_datapoint += len(tensorized_datapoint.argswap_rewrites_info.original_idxs)
        partial_minibatch["mb_num_rewrite_candidates"] += num_rewrite_candidates_in_datapoint

        partial_minibatch["mb_num_repair_groups"] += tensorized_datapoint.num_rewrite_locations_considered

        # Visualization info
        partial_minibatch["text_rewrite_original_idxs"].append(tensorized_datapoint.text_rewrites_info.original_idxs)
        partial_minibatch["varswap_original_idxs"].append(tensorized_datapoint.varswap_rewrites_info.original_idxs)
        partial_minibatch["argswap_original_idxs"].append(tensorized_datapoint.argswap_rewrites_info.original_idxs)

        # Rewrite logprobs
        partial_minibatch["rewrite_to_sample_id"].extend([current_sample_idx] * num_rewrite_candidates_in_datapoint)
        logprobs = tensorized_datapoint.observed_rewrite_logprobs
        if logprobs is not None:
            partial_minibatch["observed_rewrite_logprobs"].extend(logprobs[:-1])
            partial_minibatch["no_bug_observed_rewrite_logprobs"].append(logprobs[-1])

        return True

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        max_len = max(len(t) for t in accumulated_minibatch_data["input_subtoken_ids"])
        max_subtokens = self._token_embedder.max_num_subtokens
        num_samples = len(accumulated_minibatch_data["input_subtoken_ids"])
        input_sequence_ids = np.zeros((num_samples, max_len, max_subtokens), dtype=np.int32)
        input_seq_num_subtokens = np.ones((num_samples, max_len), dtype=np.int32)
        token_seq_lengths = np.empty(num_samples, dtype=np.int32)

        for i, seq in enumerate(accumulated_minibatch_data["input_subtoken_ids"]):
            seq_len = len(seq)
            token_seq_lengths[i] = seq_len
            for j, subtokens in enumerate(seq):
                num_subtokens = min(len(subtokens), max_subtokens)
                input_sequence_ids[i, j, :num_subtokens] = subtokens[:num_subtokens]
                input_seq_num_subtokens[i, j] = num_subtokens

        def make_tensor(data, dtype, empty_shape):
            if len(data) > 0:
                return torch.tensor(data, dtype=dtype, device=device)
            else:
                return torch.zeros(size=empty_shape, dtype=dtype, device=device)

        minibatch = {
            "input_sequence_ids": torch.tensor(input_sequence_ids, dtype=torch.int64, device=device),
            "input_seq_num_subtokens": torch.tensor(input_seq_num_subtokens, dtype=torch.int64, device=device),
            "token_sequence_lengths": torch.tensor(token_seq_lengths, dtype=torch.int64, device=device),
            "edges": torch.tensor(accumulated_minibatch_data["edges"], dtype=torch.int64, device=device),
            "edge_types": torch.tensor(accumulated_minibatch_data["edge_types"], dtype=torch.int64, device=device),
            "sample_has_bug": torch.tensor(
                accumulated_minibatch_data["sample_has_bug"], dtype=torch.bool, device=device
            ),
            "sample_to_correct_loc_idx": torch.tensor(
                accumulated_minibatch_data["sample_to_correct_loc_idx"], dtype=torch.int64, device=device
            ),
            "rewritable_loc_idxs": torch.tensor(
                accumulated_minibatch_data["rewritable_loc_idxs"], dtype=torch.int64, device=device
            ),
            "node_mappings": accumulated_minibatch_data["node_mappings"],
            # Text Rewrite
            "text_rewrite_tok_idxs": make_tensor(
                accumulated_minibatch_data["text_rewrite_tok_idxs"], dtype=torch.int64, empty_shape=(0, 2)
            ),
            "text_rewrite_replacement_ids": make_tensor(
                accumulated_minibatch_data["text_rewrite_replacement_ids"], dtype=torch.int64, empty_shape=(0,)
            ),
            "text_rewrite_to_loc_group": make_tensor(
                accumulated_minibatch_data["text_rewrite_to_loc_group"], dtype=torch.int64, empty_shape=(0,)
            ),
            "text_rewrite_correct_idxs": make_tensor(
                accumulated_minibatch_data["text_rewrite_correct_idxs"], dtype=torch.int64, empty_shape=(0,)
            ),
            "text_rewrite_joint_idxs": torch.tensor(
                accumulated_minibatch_data["text_rewrite_joint_idxs"], dtype=torch.int64, device=device
            ),
            # Var Misuse
            "varswap_tok_idxs": make_tensor(
                accumulated_minibatch_data["varswap_tok_idxs"], dtype=torch.int64, empty_shape=(0, 2)
            ),
            "varswap_replacement_tok_idxs": make_tensor(
                accumulated_minibatch_data["varswap_replacement_tok_idxs"], dtype=torch.int64, empty_shape=(0, 2)
            ),
            "varswap_to_loc_group": make_tensor(
                accumulated_minibatch_data["varswap_to_loc_group"], dtype=torch.int64, empty_shape=(0,)
            ),
            "varswap_correct_idxs": make_tensor(
                accumulated_minibatch_data["varswap_correct_idxs"], dtype=torch.int64, empty_shape=(0,)
            ),
            "varswap_joint_idxs": torch.tensor(
                accumulated_minibatch_data["varswap_joint_idxs"], dtype=torch.int64, device=device
            ),
            # Arg Swap
            "argswap_tok_idxs": make_tensor(
                accumulated_minibatch_data["argswap_tok_idxs"], dtype=torch.int64, empty_shape=(0, 2)
            ),
            "argswap_replacement_tok_idxs": make_tensor(
                accumulated_minibatch_data["argswap_replacement_tok_idxs"], dtype=torch.int64, empty_shape=(0, 3)
            ),
            "argswap_to_loc_group": make_tensor(
                accumulated_minibatch_data["argswap_to_loc_group"], dtype=torch.int64, empty_shape=(0,)
            ),
            "argswap_correct_idxs": make_tensor(
                accumulated_minibatch_data["argswap_correct_idxs"], dtype=torch.int64, empty_shape=(0,)
            ),
            "argswap_joint_idxs": torch.tensor(
                accumulated_minibatch_data["argswap_joint_idxs"], dtype=torch.int64, device=device
            ),
            # Visualisation info
            "text_rewrite_original_idxs": accumulated_minibatch_data["text_rewrite_original_idxs"],
            "varswap_original_idxs": accumulated_minibatch_data["varswap_original_idxs"],
            "argswap_original_idxs": accumulated_minibatch_data["argswap_original_idxs"],
            # Rewrite logprobs
            "rewrite_to_sample_id": torch.tensor(
                accumulated_minibatch_data["rewrite_to_sample_id"], dtype=torch.int64, device=device
            ),
        }

        # Rewrite logprobs
        if accumulated_minibatch_data["observed_rewrite_logprobs"]:
            observed_rewrite_logprobs = (
                accumulated_minibatch_data["observed_rewrite_logprobs"]
                + accumulated_minibatch_data["no_bug_observed_rewrite_logprobs"]
            )
            minibatch["observed_rewrite_logprobs"] = torch.tensor(
                observed_rewrite_logprobs, dtype=torch.float32, device=device
            )
            minibatch["rewrite_to_sample_id"] = torch.tensor(
                accumulated_minibatch_data["rewrite_to_sample_id"], dtype=torch.int64, device=device
            )
        return minibatch
