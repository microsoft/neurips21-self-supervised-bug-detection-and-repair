import math
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
from ptgnn.neuralmodels.embeddings.strelementrepresentationmodel import (
    StrElementRepresentationModel,
    SubtokenUnitEmbedder,
)

from buglab.models.layers.relational_transformer import RelationalTransformerEncoderLayer


class GreatRawDataPoint(TypedDict):
    source_tokens: List[str]
    edges: List[Tuple[int, int, int, str]]

    repair_candidates: List[Union[int, str]]  # str, seems to be only when has_bug=False

    # Ground-truth
    has_bug: bool
    error_location: int  # 0 on no-bug
    repair_targets: List[int]  # All target repair tokens, subset

    # Metadata
    bug_kind: int  ## Always 1?
    bug_kind_name: str  ## Always "VARIABLE_MISUSE"
    provenances: Any


class TensorizedGreatDataPoint(NamedTuple):
    tokens: Any

    edges: np.ndarray  # [numEdges, 2]
    edge_types: np.ndarray  # [numEdges]

    error_location: int
    repair_candidates_mask: np.ndarray  # [tokens_length]
    repair_targets_mask: np.ndarray  # [tokens_length]


class GreatVarMisuseModule(ModuleWithMetrics):
    def __init__(
        self,
        token_embedder: SubtokenUnitEmbedder,
        num_edge_types: int,
        num_layers: int,
        num_heads: int,
        intermediate_dimension: int,
        dropout_rate: float,
        rezero_mode: Literal["off", "scalar", "vector"] = "off",
        normalization_mode: Literal["prenorm", "postnorm", "off"] = "prenorm",
    ):
        super().__init__()
        embedding_dim = token_embedder.embedding_layer.embedding_dim

        # Fixed positional encodings. 5000 in GREAT implementation
        encoded_vec = np.array(
            [pos / np.power(10000, 2 * i / embedding_dim) for pos in range(5000) for i in range(embedding_dim)]
        )
        encoded_vec[::2] = np.sin(encoded_vec[::2])
        encoded_vec[1::2] = np.cos(encoded_vec[1::2])
        self.positional_encodings = nn.Parameter(
            torch.tensor(encoded_vec.reshape([1, 5000, embedding_dim]), dtype=torch.float32), requires_grad=False
        )

        self.__token_embedder = token_embedder

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
                    use_edge_value_biases=False,  # GREAT
                    rezero_mode=rezero_mode,
                    normalisation_mode=normalization_mode,
                )
                for _ in range(num_layers)
            ]
        )

        # Output layers
        self.__ln_out = nn.LayerNorm(embedding_dim)
        self.__predictions = nn.Linear(embedding_dim, 2)

    def _reset_module_metrics(self) -> None:
        self.__localization_correct = 0
        self.__localization_loss_sum = 0.0
        self.__num_samples = 0
        self.__num_correctly_localized_buggy_samples = 0

        self.__repair_correct = 0
        self.__repair_loss = 0.0
        self.__num_buggy_samples = 0

    def _module_metrics(self) -> Dict[str, Any]:
        return {
            "Localization Accuracy": self.__localization_correct / self.__num_samples,
            "Localization Accuracy (Buggy)": self.__num_correctly_localized_buggy_samples
            / (self.__num_buggy_samples + 1e-10),
            "Localization Accuracy (NoBug)": (
                self.__localization_correct - self.__num_correctly_localized_buggy_samples
            )
            / (self.__num_samples - self.__num_buggy_samples + 1e-10),
            "Repair Accuracy": self.__repair_correct / self.__num_buggy_samples,
            "Localization Loss": self.__localization_loss_sum / self.__num_samples,
            "Repair Loss": self.__repair_loss / self.__num_buggy_samples,
            "Num samples": self.__num_samples,
        }

    def forward(
        self,
        token_ids,
        token_lengths,
        input_sequence_lengths,
        edges,
        edge_types,
        error_locations,
        repair_candidates_mask,
        repair_targets_mask,
    ):
        """
        :param input_sequence:
        :param input_sequence_lengths: [BatchSize]
        :param edges: [NumEdges, 3]
        :param edge_types: [NumEdges]
        :param error_locations: [BatchSize]
        :param repair_candidates_mask: [BatchSize. MaxLength] True at candidate locations
        :param repair_targets_mask: [BatchSize, MaxLength] True at the target locations
        """
        is_buggy, localization_logits, pointer_logprobs = self.compute_logprobs(
            edge_types, edges, error_locations, token_ids, token_lengths, input_sequence_lengths, repair_candidates_mask
        )

        # To localize, no mask is added. Any token could be a candidate
        localization_loss = nn.functional.cross_entropy(localization_logits, error_locations)

        with torch.no_grad():
            localization_corrects = localization_logits.argmax(dim=-1) == error_locations
            self.__localization_correct += int(localization_corrects.sum())
            self.__num_correctly_localized_buggy_samples += int((is_buggy * localization_corrects).sum())

            self.__localization_loss_sum += float(localization_loss) * localization_corrects.shape[0]
            self.__num_samples += is_buggy.shape[0]

            num_buggy = is_buggy.sum()
            self.__num_buggy_samples += int(num_buggy)

        if num_buggy > 0:
            # Repair
            pointer_logprobs = pointer_logprobs[is_buggy]
            buggy_targets = repair_targets_mask[is_buggy]

            repair_logprobs = torch.logsumexp(pointer_logprobs.masked_fill(~buggy_targets, -math.inf), dim=-1)

            repair_loss = -repair_logprobs.mean()
            with torch.no_grad():
                repair_corrects = buggy_targets[
                    torch.arange(buggy_targets.shape[0], device=repair_targets_mask.device),
                    pointer_logprobs.argmax(dim=-1),
                ]
                self.__repair_correct += int(repair_corrects.sum())
                self.__repair_loss += float(repair_loss * num_buggy)
        else:
            repair_loss = 0

        return localization_loss + repair_loss

    def compute_logprobs(
        self,
        edge_types,
        edges,
        error_locations,
        token_ids,
        token_lengths,
        input_sequence_lengths,
        repair_candidates_mask,
    ):
        embedded_input_sequence = self.__token_embedder(
            token_idxs=token_ids.reshape(-1, token_ids.shape[-1]), lengths=token_lengths.reshape(-1)
        ).reshape(
            token_ids.shape[0], token_ids.shape[1], self.__token_embedder.embedding_layer.embedding_dim
        )  # [BatchSize, MaxLength, EmbeddingDim]

        embedded_input_sequence = (
            embedded_input_sequence + self.positional_encodings[:, : embedded_input_sequence.shape[1]]
        )

        state = embedded_input_sequence
        token_range = torch.arange(embedded_input_sequence.shape[1], device=embedded_input_sequence.device)
        token_mask = token_range.unsqueeze(0) > input_sequence_lengths.unsqueeze(-1)  # [BatchSize, MaxLength]
        for layer in self.__seq_layers:
            state = layer(state, token_mask, edges, edge_types)

        predictions_logits = self.__predictions(self.__ln_out(state))  # [BatchSize, MaxLength, 2]
        predictions_logits.masked_fill_(token_mask.unsqueeze(-1), -math.inf)

        # Localization
        localization_logits = predictions_logits[:, :, 0]

        # Repair
        is_buggy = error_locations != 0
        pointer_logits = predictions_logits[:, :, 1]
        pointer_logits.masked_fill_(~repair_candidates_mask, -math.inf)
        pointer_logprobs = torch.log_softmax(pointer_logits, dim=-1)  # [BatchSize, MaxLength]

        return is_buggy, localization_logits, pointer_logprobs


class GreatVarMisuse(AbstractNeuralModel[GreatRawDataPoint, TensorizedGreatDataPoint, GreatVarMisuseModule]):
    def __init__(
        self,
        transformer_config: Dict[str, Any],
        vocab_size: int,
        embedding_dim: int,
        max_length: int = 512,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.__transformer_config = transformer_config
        self.__max_length = max_length

        self.__token_embedder = StrElementRepresentationModel(
            token_splitting="subtoken",
            embedding_size=embedding_dim,
            dropout_rate=dropout_rate,
            vocabulary_size=vocab_size,
            subtoken_combination="mean",
        )

    def initialize_metadata(self) -> None:
        self.edges = set()

    def update_metadata_from(self, datapoint: GreatRawDataPoint) -> None:
        for token in datapoint["source_tokens"]:
            self.__token_embedder.update_metadata_from(token)

        for _, _, edge_id, _ in datapoint["edges"]:
            self.edges.add(edge_id)

    def finalize_metadata(self) -> None:
        self.__edge_id_to_edge = {e: i for i, e in enumerate(self.edges)}

    def build_neural_module(self) -> GreatVarMisuseModule:
        return GreatVarMisuseModule(
            token_embedder=self.__token_embedder.build_neural_module(),
            num_edge_types=2 * len(self.__edge_id_to_edge),
            **self.__transformer_config
        )

    def tensorize(self, datapoint: GreatRawDataPoint) -> Optional[TensorizedGreatDataPoint]:
        token_sequence_length = len(datapoint["source_tokens"])
        if token_sequence_length > self.__max_length:
            return None

        error_location = datapoint["error_location"]
        if error_location > 0:
            repair_candidates_mask = np.zeros(token_sequence_length, dtype=np.bool)
            repair_targets_mask = np.zeros(token_sequence_length, dtype=np.bool)

            for repair_candidate in datapoint["repair_candidates"]:
                repair_candidates_mask[repair_candidate] = True
            for repair_target in datapoint["repair_targets"]:
                repair_targets_mask[repair_target] = True

            if not np.any(repair_candidates_mask & repair_targets_mask):
                return None
        else:
            repair_candidates_mask = None
            repair_targets_mask = None

        return TensorizedGreatDataPoint(
            tokens=[self.__token_embedder.tensorize(t) for t in datapoint["source_tokens"]],
            edges=np.array([[e1, e2] for e1, e2, _, _ in datapoint["edges"]], dtype=np.int32),
            edge_types=np.array(
                [self.__edge_id_to_edge[etype] for _, _, etype, _ in datapoint["edges"]], dtype=np.int32
            ),
            error_location=error_location,
            repair_candidates_mask=repair_candidates_mask,
            repair_targets_mask=repair_targets_mask,
        )

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {"samples": []}

    def extend_minibatch_with(
        self, tensorized_datapoint: TensorizedGreatDataPoint, partial_minibatch: Dict[str, Any]
    ) -> bool:
        partial_minibatch["samples"].append(tensorized_datapoint)
        return True

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        samples: List[TensorizedGreatDataPoint] = accumulated_minibatch_data["samples"]
        input_sequence_lengths = np.array([len(s.tokens) for s in samples], dtype=np.int32)
        max_seq_size = int(input_sequence_lengths.max())
        max_num_subtokens = self.__token_embedder.max_num_subtokens

        token_idxs = np.zeros((len(samples), max_seq_size, max_num_subtokens), dtype=np.int32)
        token_lengths = np.ones((len(samples), max_seq_size), dtype=np.int32)
        error_locations = np.zeros(len(samples), dtype=np.int32)
        repair_candidates_mask = np.zeros((len(samples), max_seq_size), dtype=np.bool)
        repair_targets_mask = np.zeros((len(samples), max_seq_size), dtype=np.bool)
        edges = []
        edge_types = []
        for i, sample in enumerate(samples):
            seq_size = len(sample.tokens)
            for j, token in enumerate(sample.tokens):
                lim = min(len(token), max_num_subtokens)
                token_idxs[i, j, :lim] = token[:lim]
                token_lengths[i, j] = lim

            error_locations[i] = sample.error_location
            if sample.error_location > 0:
                repair_candidates_mask[i, :seq_size] = sample.repair_candidates_mask
                repair_targets_mask[i, :seq_size] = sample.repair_targets_mask

            edges.append(np.concatenate([np.full((len(sample.edges), 1), fill_value=i), sample.edges], axis=-1))
            edge_types.append(sample.edge_types)

        edges = np.concatenate(edges)
        edge_types = np.concatenate(edge_types)

        reverse_edges = np.stack([edges[:, 0], edges[:, 2], edges[:, 1]], axis=-1)
        edges = np.concatenate([edges, reverse_edges], axis=0)
        edge_types = np.concatenate([edge_types, edge_types + len(self.__edge_id_to_edge)], axis=0)

        return {
            "token_ids": torch.tensor(token_idxs, dtype=torch.int64, device=device),
            "token_lengths": torch.tensor(token_lengths, dtype=torch.int32, device=device),
            "input_sequence_lengths": torch.tensor(input_sequence_lengths, dtype=torch.int32, device=device),
            "edges": torch.tensor(edges, dtype=torch.int64, device=device),
            "edge_types": torch.tensor(edge_types, dtype=torch.int64, device=device),
            "error_locations": torch.tensor(error_locations, dtype=torch.int64, device=device),
            "repair_candidates_mask": torch.tensor(repair_candidates_mask, dtype=torch.bool, device=device),
            "repair_targets_mask": torch.tensor(repair_targets_mask, dtype=torch.bool, device=device),
        }
