import logging
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
from ptgnn.neuralmodels.embeddings.strelementrepresentationmodel import (
    CharUnitEmbedder,
    StrElementRepresentationModel,
    SubtokenUnitEmbedder,
    TokenUnitEmbedder,
)
from torch import nn
from typing_extensions import Literal

from buglab.models.basemodel import AbstractBugLabModel
from buglab.models.layers.fixermodules import (
    CandidatePairSelectorModule,
    SingleCandidateNodeSelectorModule,
    TextRepairModule,
)
from buglab.models.layers.localizationmodule import LocalizationModule
from buglab.models.layers.relational_transformer import RelationalTransformerEncoderLayer
from buglab.models.utils import compute_generator_loss, scatter_log_softmax
from buglab.representations.data import BugLabData, BugLabGraph

LOGGER = logging.getLogger(__name__)


class SeqModelTensorizedSample(NamedTuple):
    target_subtokens_ids: List[int]
    intra_token_edges: Dict[str, List[Tuple[int, int]]]
    candidate_location_idxs: List[int]

    target_location_idx: Optional[int]

    node_mappings: Dict[int, int]

    # Bug repair
    target_rewrite_node_ids: List[int]
    target_rewrites: List[int]
    target_rewrite_to_location_group: List[int]
    correct_rewrite_target: Optional[int]
    text_rewrite_original_idx: List[int]

    varmisused_node_ids: List[int]
    candidate_symbol_node_ids: List[int]
    candidate_symbol_to_varmisused_node: List[int]
    correct_candidate_symbol_node: Optional[int]
    candidate_rewrite_original_idx: List[int]

    call_node_ids: List[int]
    candidate_swapped_node_ids: List[int]
    swapped_pair_to_call: List[int]
    correct_swapped_pair: Optional[int]
    pair_rewrite_original_idx: List[int]

    num_rewrite_locations_considered: int

    # Bug selection
    rewrite_logprobs: Optional[List[float]]


class SeqBugLabModule(ModuleWithMetrics):
    def __init__(
        self,
        token_embedder: Union[TokenUnitEmbedder, SubtokenUnitEmbedder, CharUnitEmbedder],
        embedding_dim: int,
        num_edge_types: int,
        num_layers: int,
        num_heads: int,
        intermediate_dimension: int,
        dropout_rate: float,
        rewrite_vocabulary_size: int,
        layer_type: Literal["great", "rat", "transformer", "gru"] = "great",
        buggy_samples_weight_schedule: Callable[[int], float] = lambda _: 1.0,
        generator_loss_type: Optional[str] = "norm-kl",
        rezero_mode: Literal["off", "scalar", "vector"] = "off",
        normalisation_mode: Literal["off", "prenorm", "postnorm"] = "postnorm",
    ):
        super().__init__()
        self.__generator_loss_type = generator_loss_type
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

        output_representation_size = embedding_dim
        self.__localization_module = LocalizationModule(
            output_representation_size, buggy_samples_weight_schedule=buggy_samples_weight_schedule
        )
        self._buggy_samples_weight_schedule = buggy_samples_weight_schedule

        self._text_repair_module = TextRepairModule(embedding_dim, rewrite_vocabulary_size)
        self._varmisuse_module = SingleCandidateNodeSelectorModule(embedding_dim)
        self._argswap_module = CandidatePairSelectorModule(embedding_dim)

    def _reset_module_metrics(self) -> None:
        if not hasattr(self, "_epoch_idx"):
            self._epoch_idx = 0
        elif self.training and self.__num_batches > 0:
            # Assumes that module metrics are reset once per epoch.
            self._epoch_idx += 1

        self.__loss = 0.0
        self.__repair_loss = 0.0

        self.__total_samples = 0
        self.__num_batches = 0

    def _module_metrics(self) -> Dict[str, Any]:
        metrics = {}
        if self.__total_samples > 0:
            metrics["Repair Loss"] = self.__repair_loss / self.__total_samples
        if self.__num_batches > 0:
            metrics["Loss"] = self.__loss / self.__num_batches
        return metrics

    def _compute_repair_logprobs(
        self,
        output_representations,
        target_rewrite_node_ids,
        target_rewrites,
        rewrite_to_location_group,
        varmisused_node_ids,
        candidate_symbol_node_ids,
        candidate_symbol_to_location_group,
        call_node_ids,
        candidate_swapped_node_ids,
        swapped_pair_to_call_location_group,
    ):
        if target_rewrites.shape[0] > 0:
            text_repair_logits = self._text_repair_module.compute_rewrite_logits(
                target_rewrite_node_representations=output_representations[
                    target_rewrite_node_ids[:, 0], target_rewrite_node_ids[:, 1]
                ],
                candidate_rewrites=target_rewrites,
            )
        else:
            text_repair_logits = torch.zeros(0, device=target_rewrites.device)

        if varmisused_node_ids.shape[0] > 0:
            varmisuse_logits = self._varmisuse_module.compute_per_slot_log_probability(
                slot_representations_per_target=output_representations[
                    varmisused_node_ids[:, 0], varmisused_node_ids[:, 1]
                ],
                target_nodes_representations=output_representations[
                    candidate_symbol_node_ids[:, 0], candidate_symbol_node_ids[:, 1]
                ],
            )
        else:
            varmisuse_logits = torch.zeros(0, device=varmisused_node_ids.device)

        if call_node_ids.shape[0] > 0:
            arg_swap_logits = self._argswap_module.compute_per_pair_logits(
                slot_representations_per_pair=output_representations[call_node_ids[:, 0], call_node_ids[:, 1]],
                pair_representations=output_representations[
                    candidate_swapped_node_ids[:, 0].unsqueeze(-1), candidate_swapped_node_ids[:, 1:]
                ],
            )
        else:
            arg_swap_logits = torch.zeros(0, device=call_node_ids.device)

        all_logits = torch.cat((text_repair_logits, varmisuse_logits, arg_swap_logits))
        logit_groups = torch.cat(
            (rewrite_to_location_group, candidate_symbol_to_location_group, swapped_pair_to_call_location_group)
        )
        logprobs = scatter_log_softmax(all_logits, index=logit_groups)
        text_repair_logprobs, varmisuse_logprobs, arg_swap_logprobs = torch.split(
            logprobs, [text_repair_logits.shape[0], varmisuse_logits.shape[0], arg_swap_logits.shape[0]]
        )
        return arg_swap_logprobs, text_repair_logprobs, varmisuse_logprobs

    def _compute_localization_logprobs(self, candidate_reprs, candidate_to_sample_idx, num_samples):
        candidate_to_sample_idx, candidate_log_probs, _ = self.__localization_module.compute_localization_logprobs(
            candidate_reprs, candidate_to_sample_idx, num_samples
        )
        return candidate_to_sample_idx, candidate_log_probs

    def forward(
        self,
        *,
        input_sequence_ids,
        input_seq_num_subtokens,
        token_sequence_lengths,
        edges,
        edge_types,
        has_bug,
        candidate_location_idxs,
        target_location_idxs,
        # Text repair
        target_rewrite_node_ids: torch.IntTensor,
        target_rewrites: torch.IntTensor,
        rewrite_to_location_group: torch.IntTensor,
        correct_rewrite_idxs: torch.IntTensor,
        text_rewrite_idxs: torch.IntTensor,
        # Var Misuse
        varmisused_node_ids: torch.IntTensor,
        candidate_symbol_node_ids: torch.IntTensor,
        candidate_symbol_to_location_group: torch.IntTensor,
        correct_candidate_symbols: torch.IntTensor,
        candidate_rewrite_idxs: torch.IntTensor,
        # ArgSwap
        call_node_ids: torch.IntTensor,
        candidate_swapped_node_ids: torch.IntTensor,
        swapped_pair_to_call_location_group: torch.IntTensor,
        correct_swapped_pair: torch.IntTensor,
        pair_rewrite_idxs: torch.IntTensor,
        # Rewrite logprobs
        rewrite_to_graph_id: Optional[torch.IntTensor] = None,
        rewrite_logprobs: Optional[torch.FloatTensor] = None,
        **kwargs,  # Visualization data (ignored)
    ):
        """
        :param input_sequence_ids: [B, max_seq_len]
        :param input_sequence_lengths: The size of the input sequence [B]
        :param token_sequence_lengths: The size of the tokens in the input sequence (excluding symbols) [B]
        :param edges: [num_edges, 3]
        :param edge_types: [num_edges]
        :param has_bug: [B]
        :param candidate_location_idxs: [num_target_locations, 2]
        :param target_location_idxs: [B]
        """
        output_representation = self._compute_output_representation(
            input_sequence_ids, input_seq_num_subtokens, token_sequence_lengths, edges, edge_types
        )

        target_location_representations = output_representation[
            candidate_location_idxs[:, 0], candidate_location_idxs[:, 1]
        ]  # [num_candidate_points, H]

        # Repair
        arg_swap_logprobs, text_repair_logprobs, varmisuse_logprobs = self._compute_repair_logprobs(
            output_representation,
            target_rewrite_node_ids,
            target_rewrites,
            rewrite_to_location_group,
            varmisused_node_ids,
            candidate_symbol_node_ids,
            candidate_symbol_to_location_group,
            call_node_ids,
            candidate_swapped_node_ids,
            swapped_pair_to_call_location_group,
        )

        if rewrite_logprobs is not None:
            candidate_to_sample_idx = candidate_location_idxs[:, 0]
            loss_type = self.__generator_loss_type
            # Note: the command
            # torch.exp(torch_scatter.scatter_logsumexp(localization_logprobs, torch.cat((candidate_to_sample_idx, arange))))
            # should give a tensor of ones here (total probabilities).
            _, localization_logprobs, arrange, = self.__localization_module.compute_localization_logprobs(
                candidate_reprs=target_location_representations,
                candidate_to_sample_idx=candidate_to_sample_idx,
                num_samples=has_bug.shape[0],
            )

            loss = compute_generator_loss(
                arg_swap_logprobs,
                arrange,
                candidate_rewrite_idxs,
                candidate_symbol_to_location_group,
                localization_logprobs,
                loss_type,
                pair_rewrite_idxs,
                rewrite_logprobs,
                rewrite_to_graph_id,
                rewrite_to_location_group,
                swapped_pair_to_call_location_group,
                text_repair_logprobs,
                text_rewrite_idxs,
                varmisuse_logprobs,
            )

            with torch.no_grad():
                self.__loss += float(loss)
                self.__num_batches += 1

            return loss

        else:
            localization_loss = self.__localization_module(
                candidate_reprs=target_location_representations,
                candidate_to_sample_idx=candidate_location_idxs[:, 0],
                has_bug=has_bug,
                correct_candidate_idxs=target_location_idxs,
            )

            text_repair_loss = self._text_repair_module(text_repair_logprobs, correct_rewrite_idxs)
            varmisuse_repair_loss = self._varmisuse_module(varmisuse_logprobs, correct_candidate_symbols)
            arg_swap_repair_loss = self._argswap_module(arg_swap_logprobs, correct_swapped_pair)

            buggy_samples_weight = self._buggy_samples_weight_schedule(self._epoch_idx)
            repair_loss = text_repair_loss.sum() + varmisuse_repair_loss.sum() + arg_swap_repair_loss.sum()
            repair_loss = repair_loss * buggy_samples_weight

            with torch.no_grad():
                self.__total_samples += float(has_bug.sum().cpu())
                self.__repair_loss += float(repair_loss)

                self.__num_batches += 1
                self.__loss += float(localization_loss + repair_loss / has_bug.shape[0])

            return localization_loss + repair_loss / has_bug.shape[0]

    def _compute_output_representation(
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
        buggy_samples_weight_schedule: Callable[[int], float] = lambda _: 1.0,
        generator_loss_type: Optional[str] = "classify-max-loss",
        rezero_mode: Literal["off", "scalar", "vector"] = "off",
        normalisation_mode: Literal["off", "prenorm", "postnorm"] = "postnorm",
    ):
        super().__init__()
        self._init()
        self.__tensorize_only_at_target_location_rewrites = True
        self.__edge_types = set()

        self.__dropout_rate = dropout_rate
        self.__representation_size = representation_size
        self.__layer_type = layer_type

        self.__max_seq_size = max_seq_size
        self.__token_embedder = StrElementRepresentationModel(
            token_splitting="subtoken",
            embedding_size=representation_size,
            dropout_rate=dropout_rate,
            vocabulary_size=max_subtoken_vocab_size,
            subtoken_combination="max",
        )

        self.__buggy_samples_weight_schedule = buggy_samples_weight_schedule

        self.__num_heads = num_heads
        self.__num_layers = num_layers
        self.__intermediate_dimension_size = intermediate_dimension_size
        self.__generator_loss_type = generator_loss_type
        self.__rezero_mode = rezero_mode
        self.__normalisation_mode = normalisation_mode

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
            LOGGER.error(f"Encountered graph where the tokens are not connected in %s", graph["path"])
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
        except:
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
            self.__token_embedder.update_metadata_from(n)

        self.__edge_types.update(edges.keys())

    def finalize_metadata(self) -> None:
        self.__edge_types = list(self.__edge_types)
        self.__edge_type_to_idx = {t: i for i, t in enumerate(self.__edge_types)}

    def build_neural_module(self) -> SeqBugLabModule:
        return SeqBugLabModule(
            token_embedder=self.__token_embedder.build_neural_module(),
            embedding_dim=self.__token_embedder.embedding_size,
            num_edge_types=len(self.__edge_types),
            num_layers=self.__num_layers,
            num_heads=self.__num_heads,
            intermediate_dimension=self.__intermediate_dimension_size,
            dropout_rate=self.__dropout_rate,
            rewrite_vocabulary_size=len(self._target_rewrite_ops),
            layer_type=self.__layer_type,
            buggy_samples_weight_schedule=self.__buggy_samples_weight_schedule,
            generator_loss_type=self.__generator_loss_type,
            rezero_mode=self.__rezero_mode,
            normalisation_mode=self.__normalisation_mode,
        )

    def tensorize(self, datapoint: BugLabData) -> Optional[SeqModelTensorizedSample]:

        if "candidate_rewrite_logprobs" in datapoint:
            assert not self._tensorize_only_at_target_location_rewrites

        try:
            token_data = self.__to_token_data(datapoint["graph"])
        except Exception as ex:
            LOGGER.exception("Error in generating token sequence for %s", datapoint["graph"]["path"], exc_info=ex)
            return None
        if token_data is None:
            return None
        node_labels, old_node_id_to_new, edges, reference_nodes = token_data

        if len(node_labels) > self.__max_seq_size:
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
            # Rewrites
            target_rewrite_node_ids,
            target_rewrites,
            target_rewrite_to_location_group,
            correct_rewrite_target,
            text_rewrite_original_idx,
            # VarMisuse
            varmisused_node_ids,
            candidate_symbol_to_varmisused_location,
            candidate_symbol_node_ids,
            correct_candidate_symbol_node,
            varmisuse_rewrite_original_idx,
            # Arg Swap
            call_node_ids,
            candidate_swapped_node_ids,
            correct_swapped_pair,
            swapped_pair_to_call,
            swapped_rewrite_original_ids,
            # All rewrites
            repr_location_group_ids,
        ) = self._compute_rewrite_data(datapoint, candidate_node_idxs)

        target_rewrite_node_ids = [old_node_id_to_new[n] for n in target_rewrite_node_ids]
        varmisused_node_ids = [old_node_id_to_new[n] for n in varmisused_node_ids]
        candidate_symbol_node_ids = [old_node_id_to_new[n] for n in candidate_symbol_node_ids]
        call_node_ids = [old_node_id_to_new[n] for n in call_node_ids]
        candidate_swapped_node_ids = [
            (old_node_id_to_new[p1], old_node_id_to_new[p2]) for p1, p2 in candidate_swapped_node_ids
        ]

        return SeqModelTensorizedSample(
            target_subtokens_ids=[self.__token_embedder.tensorize(t) for t in node_labels],
            intra_token_edges=edges,
            candidate_location_idxs=transformed_candidate_node_idxs,
            target_location_idx=target_node_idx,
            node_mappings=old_node_id_to_new,
            ### Bug repair
            # Text rewrite
            target_rewrite_node_ids=target_rewrite_node_ids,
            target_rewrites=target_rewrites,
            target_rewrite_to_location_group=target_rewrite_to_location_group,
            correct_rewrite_target=correct_rewrite_target,
            text_rewrite_original_idx=text_rewrite_original_idx,
            # Var Misuse
            varmisused_node_ids=varmisused_node_ids,
            candidate_symbol_node_ids=candidate_symbol_node_ids,
            candidate_symbol_to_varmisused_node=candidate_symbol_to_varmisused_location,
            correct_candidate_symbol_node=correct_candidate_symbol_node,
            candidate_rewrite_original_idx=varmisuse_rewrite_original_idx,
            # Arg Swap
            call_node_ids=call_node_ids,
            candidate_swapped_node_ids=candidate_swapped_node_ids,
            swapped_pair_to_call=swapped_pair_to_call,
            correct_swapped_pair=correct_swapped_pair,
            pair_rewrite_original_idx=swapped_rewrite_original_ids,
            # All rewrites
            num_rewrite_locations_considered=len(repr_location_group_ids),
            # Rewrite logprobs
            rewrite_logprobs=datapoint.get("candidate_rewrite_logprobs", None),
        )

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            "input_subtoken_ids": [],
            "edges": [],
            "edge_types": [],
            "candidate_location_idxs": [],
            "has_bug": [],
            "target_location_idxs": [],
            "node_mappings": [],
            # Repair
            "mb_num_repair_groups": 0,
            "mb_num_rewrite_candidates": 0,
            # Text rewrites
            "target_rewrite_node_ids": [],
            "target_rewrites": [],
            "rewrite_to_location_group": [],
            "correct_rewrite_idxs": [],
            "text_rewrite_original_idxs": [],
            "text_rewrite_idxs": [],
            # Var Misuse
            "varmisused_node_ids": [],
            "candidate_symbol_node_ids": [],
            "candidate_symbol_to_location_group": [],
            "correct_candidate_symbols": [],
            "candidate_rewrite_original_idxs": [],
            "candidate_rewrite_idxs": [],
            # Arg Swaps
            "call_node_ids": [],
            "candidate_swapped_node_ids": [],
            "swapped_pair_to_call_location_group": [],
            "correct_swapped_pair": [],
            "pair_rewrite_original_idx": [],
            "pair_rewrite_idxs": [],
            # Rewrite logprobs
            "rewrite_to_graph_id": [],
            "rewrite_logprobs": [],
            "no_bug_rewrite_logprobs": [],
        }

    def extend_minibatch_with(
        self, tensorized_datapoint: SeqModelTensorizedSample, partial_minibatch: Dict[str, Any]
    ) -> bool:
        current_sample_idx = len(partial_minibatch["input_subtoken_ids"])
        partial_minibatch["input_subtoken_ids"].append(tensorized_datapoint.target_subtokens_ids)

        for edge_type, adj_list in tensorized_datapoint.intra_token_edges.items():
            partial_minibatch["edge_types"].extend(self.__edge_type_to_idx[edge_type] for _ in adj_list)
            partial_minibatch["edges"].extend((current_sample_idx, f, t) for f, t in adj_list)

        partial_minibatch["has_bug"].append(tensorized_datapoint.target_location_idx is not None)

        partial_minibatch["node_mappings"].append(tensorized_datapoint.node_mappings)

        target_location_idx = tensorized_datapoint.target_location_idx or 0
        partial_minibatch["target_location_idxs"].append(
            target_location_idx + len(partial_minibatch["candidate_location_idxs"])
        )
        partial_minibatch["candidate_location_idxs"].extend(
            (current_sample_idx, tid) for tid in tensorized_datapoint.candidate_location_idxs
        )

        mb_rewrite_candidate_offset = partial_minibatch["mb_num_rewrite_candidates"]
        num_rewrite_candidates_in_datapoint = 0
        mb_num_repair_groups_offset = partial_minibatch["mb_num_repair_groups"]

        # Text rewrites
        if tensorized_datapoint.correct_rewrite_target is not None:
            partial_minibatch["correct_rewrite_idxs"].append(
                tensorized_datapoint.correct_rewrite_target + len(partial_minibatch["target_rewrites"])
            )
        partial_minibatch["target_rewrite_node_ids"].extend(
            (current_sample_idx, t) for t in tensorized_datapoint.target_rewrite_node_ids
        )
        partial_minibatch["target_rewrites"].extend(tensorized_datapoint.target_rewrites)
        partial_minibatch["rewrite_to_location_group"].extend(
            t + mb_num_repair_groups_offset for t in tensorized_datapoint.target_rewrite_to_location_group
        )
        partial_minibatch["text_rewrite_idxs"].extend(
            t + mb_rewrite_candidate_offset for t in tensorized_datapoint.text_rewrite_original_idx
        )
        num_rewrite_candidates_in_datapoint += len(tensorized_datapoint.text_rewrite_original_idx)

        # Var misuse
        if tensorized_datapoint.correct_candidate_symbol_node is not None:
            partial_minibatch["correct_candidate_symbols"].append(
                tensorized_datapoint.correct_candidate_symbol_node
                + len(partial_minibatch["candidate_symbol_to_location_group"])
            )
        partial_minibatch["varmisused_node_ids"].extend(
            (current_sample_idx, t) for t in tensorized_datapoint.varmisused_node_ids
        )
        partial_minibatch["candidate_symbol_node_ids"].extend(
            (current_sample_idx, t) for t in tensorized_datapoint.candidate_symbol_node_ids
        )
        partial_minibatch["candidate_symbol_to_location_group"].extend(
            t + mb_num_repair_groups_offset for t in tensorized_datapoint.candidate_symbol_to_varmisused_node
        )
        partial_minibatch["candidate_rewrite_idxs"].extend(
            t + mb_rewrite_candidate_offset for t in tensorized_datapoint.candidate_rewrite_original_idx
        )
        num_rewrite_candidates_in_datapoint += len(tensorized_datapoint.candidate_rewrite_original_idx)

        # Arg Swaps
        if tensorized_datapoint.correct_swapped_pair is not None:
            partial_minibatch["correct_swapped_pair"].append(
                tensorized_datapoint.correct_swapped_pair
                + len(partial_minibatch["swapped_pair_to_call_location_group"])
            )
        partial_minibatch["call_node_ids"].extend((current_sample_idx, t) for t in tensorized_datapoint.call_node_ids)
        partial_minibatch["candidate_swapped_node_ids"].extend(
            (current_sample_idx, t1, t2) for t1, t2 in tensorized_datapoint.candidate_swapped_node_ids
        )
        partial_minibatch["swapped_pair_to_call_location_group"].extend(
            t + mb_num_repair_groups_offset for t in tensorized_datapoint.swapped_pair_to_call
        )
        partial_minibatch["pair_rewrite_idxs"].extend(
            t + mb_rewrite_candidate_offset for t in tensorized_datapoint.pair_rewrite_original_idx
        )
        num_rewrite_candidates_in_datapoint += len(tensorized_datapoint.pair_rewrite_original_idx)
        partial_minibatch["mb_num_rewrite_candidates"] += num_rewrite_candidates_in_datapoint

        partial_minibatch["mb_num_repair_groups"] += tensorized_datapoint.num_rewrite_locations_considered

        # Visualization info
        partial_minibatch["text_rewrite_original_idxs"].append(tensorized_datapoint.text_rewrite_original_idx)
        partial_minibatch["candidate_rewrite_original_idxs"].append(tensorized_datapoint.candidate_rewrite_original_idx)
        partial_minibatch["pair_rewrite_original_idx"].append(tensorized_datapoint.pair_rewrite_original_idx)

        # Rewrite logprobs
        partial_minibatch["rewrite_to_graph_id"].extend([current_sample_idx] * num_rewrite_candidates_in_datapoint)
        logprobs = tensorized_datapoint.rewrite_logprobs
        if logprobs is not None:
            partial_minibatch["rewrite_logprobs"].extend(logprobs[:-1])
            partial_minibatch["no_bug_rewrite_logprobs"].append(logprobs[-1])

        return True

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        max_len = max(len(t) for t in accumulated_minibatch_data["input_subtoken_ids"])
        max_subtokens = self.__token_embedder.max_num_subtokens
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

        minibatch = {
            "input_sequence_ids": torch.tensor(input_sequence_ids, dtype=torch.int64, device=device),
            "input_seq_num_subtokens": torch.tensor(input_seq_num_subtokens, dtype=torch.int64, device=device),
            "token_sequence_lengths": torch.tensor(token_seq_lengths, dtype=torch.int64, device=device),
            "edges": torch.tensor(accumulated_minibatch_data["edges"], dtype=torch.int64, device=device),
            "edge_types": torch.tensor(accumulated_minibatch_data["edge_types"], dtype=torch.int64, device=device),
            "has_bug": torch.tensor(accumulated_minibatch_data["has_bug"], dtype=torch.bool, device=device),
            "target_location_idxs": torch.tensor(
                accumulated_minibatch_data["target_location_idxs"], dtype=torch.int64, device=device
            ),
            "candidate_location_idxs": torch.tensor(
                accumulated_minibatch_data["candidate_location_idxs"], dtype=torch.int64, device=device
            ),
            "node_mappings": accumulated_minibatch_data["node_mappings"],
            # Text Rewrite
            "target_rewrite_node_ids": torch.tensor(
                accumulated_minibatch_data["target_rewrite_node_ids"], dtype=torch.int64, device=device
            ),
            "target_rewrites": torch.tensor(
                accumulated_minibatch_data["target_rewrites"], dtype=torch.int64, device=device
            ),
            "rewrite_to_location_group": torch.tensor(
                accumulated_minibatch_data["rewrite_to_location_group"], dtype=torch.int64, device=device
            ),
            "correct_rewrite_idxs": torch.tensor(
                accumulated_minibatch_data["correct_rewrite_idxs"], dtype=torch.int64, device=device
            ),
            # Var Misuse
            "varmisused_node_ids": torch.tensor(
                accumulated_minibatch_data["varmisused_node_ids"], dtype=torch.int64, device=device
            ),
            "candidate_symbol_node_ids": torch.tensor(
                accumulated_minibatch_data["candidate_symbol_node_ids"], dtype=torch.int64, device=device
            ),
            "candidate_symbol_to_location_group": torch.tensor(
                accumulated_minibatch_data["candidate_symbol_to_location_group"], dtype=torch.int64, device=device
            ),
            "correct_candidate_symbols": torch.tensor(
                accumulated_minibatch_data["correct_candidate_symbols"], dtype=torch.int64, device=device
            ),
            # Arg Swap
            "call_node_ids": torch.tensor(
                accumulated_minibatch_data["call_node_ids"], dtype=torch.int64, device=device
            ),
            "candidate_swapped_node_ids": torch.tensor(
                accumulated_minibatch_data["candidate_swapped_node_ids"], dtype=torch.int64, device=device
            ),
            "swapped_pair_to_call_location_group": torch.tensor(
                accumulated_minibatch_data["swapped_pair_to_call_location_group"], dtype=torch.int64, device=device
            ),
            "correct_swapped_pair": torch.tensor(
                accumulated_minibatch_data["correct_swapped_pair"], dtype=torch.int64, device=device
            ),
            # Visualisation info
            "text_rewrite_original_idxs": accumulated_minibatch_data["text_rewrite_original_idxs"],
            "candidate_rewrite_original_idxs": accumulated_minibatch_data["candidate_rewrite_original_idxs"],
            "pair_rewrite_original_idx": accumulated_minibatch_data["pair_rewrite_original_idx"],
            # Rewrite info
            "text_rewrite_idxs": torch.tensor(
                accumulated_minibatch_data["text_rewrite_idxs"], dtype=torch.int64, device=device
            ),
            "candidate_rewrite_idxs": torch.tensor(
                accumulated_minibatch_data["candidate_rewrite_idxs"], dtype=torch.int64, device=device
            ),
            "pair_rewrite_idxs": torch.tensor(
                accumulated_minibatch_data["pair_rewrite_idxs"], dtype=torch.int64, device=device
            ),
            # Rewrite logprobs
            "rewrite_to_graph_id": torch.tensor(
                accumulated_minibatch_data["rewrite_to_graph_id"], dtype=torch.int64, device=device
            ),
        }

        # Rewrite logprobs
        if accumulated_minibatch_data["rewrite_logprobs"]:
            rewrite_logprobs = (
                accumulated_minibatch_data["rewrite_logprobs"] + accumulated_minibatch_data["no_bug_rewrite_logprobs"]
            )
            minibatch["rewrite_logprobs"] = torch.tensor(rewrite_logprobs, dtype=torch.float32, device=device)
            minibatch["rewrite_to_graph_id"] = torch.tensor(
                accumulated_minibatch_data["rewrite_to_graph_id"], dtype=torch.int64, device=device
            )
        return minibatch

    def predict(
        self, data: Iterator[BugLabData], trained_nn: SeqBugLabModule, device, parallelize: bool
    ) -> Iterator[Tuple[BugLabData, Dict[int, float], List[float]]]:
        trained_nn.eval()
        with torch.no_grad(), self._tensorize_all_location_rewrites():
            for mb_data, original_datapoints in self.minibatch_iterator(
                self.tensorize_dataset(data, return_input_data=True, parallelize=parallelize),
                device,
                max_minibatch_size=50,
                parallelize=parallelize,
            ):
                num_samples = mb_data["input_sequence_ids"].shape[0]

                output_representation = trained_nn._compute_output_representation(
                    mb_data["input_sequence_ids"],
                    mb_data["input_seq_num_subtokens"],
                    mb_data["token_sequence_lengths"],
                    mb_data["edges"],
                    mb_data["edge_types"],
                )

                candidate_location_idxs = mb_data["candidate_location_idxs"]

                target_location_representations = output_representation[
                    candidate_location_idxs[:, 0], candidate_location_idxs[:, 1]
                ]  # [num_candidate_points, H]

                # Quick sanity check:
                # torch.exp(torch_scatter.scatter_logsumexp(candidate_log_probs, candidate_to_sample_idx))
                # should be ones.
                candidate_to_sample_idx, candidate_log_probs = trained_nn._compute_localization_logprobs(
                    target_location_representations, candidate_location_idxs[:, 0], num_samples
                )

                # Repair
                arg_swap_logprobs, text_repair_logprobs, varmisuse_logprobs = trained_nn._compute_repair_logprobs(
                    output_representation,
                    mb_data["target_rewrite_node_ids"],
                    mb_data["target_rewrites"],
                    mb_data["rewrite_to_location_group"],
                    mb_data["varmisused_node_ids"],
                    mb_data["candidate_symbol_node_ids"],
                    mb_data["candidate_symbol_to_location_group"],
                    mb_data["call_node_ids"],
                    mb_data["candidate_swapped_node_ids"],
                    mb_data["swapped_pair_to_call_location_group"],
                )

                candidate_to_sample_idx = candidate_to_sample_idx.cpu().numpy()
                candidate_log_probs = candidate_log_probs.cpu().numpy()

                yield from self._iter_per_sample_results(
                    mb_data,
                    candidate_to_sample_idx,
                    candidate_log_probs,
                    arg_swap_logprobs,
                    num_samples,
                    original_datapoints,
                    text_repair_logprobs,
                    varmisuse_logprobs,
                    node_mappings=mb_data["node_mappings"],
                )
