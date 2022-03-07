from typing import Any, Dict, Final, NamedTuple, Union

import numpy as np
import torch
from ptgnn.baseneuralmodel import AbstractNeuralModel
from ptgnn.neuralmodels.embeddings.strelementrepresentationmodel import StrElementRepresentationModel
from ptgnn.neuralmodels.gnn import AbstractNodeEmbedder
from torch import nn


class TensorizedStr(NamedTuple):
    data: Any
    is_positional: bool


class PosEncodingMixedEmbedder(nn.Module):
    def __init__(self, embedding_size: int, base_embedder: nn.Module, learnt_embeddings=False):
        super().__init__()
        self.base_embedder = base_embedder

        assert embedding_size % 2 == 0, "Positional encodings require even number of embedding size"
        if learnt_embeddings:
            self.positional_encoding = nn.Parameter(torch.randn(10000, embedding_size), requires_grad=True)
        else:
            periods = 1.0 / 10000 ** (2 * torch.arange(embedding_size // 2) / embedding_size)
            self.periods = nn.Parameter(periods.unsqueeze(0), requires_grad=False)
        self.learnt_embeddings = learnt_embeddings

    def forward(self, base_model_tokens, positional_tokens, base_idxs_of_tokens):
        embedded_tokens = self.base_embedder(**base_model_tokens)
        if self.learnt_embeddings:
            positional_embeddings = self.positional_encoding[positional_tokens]
        else:
            with torch.no_grad():
                positions = self.periods * positional_tokens.unsqueeze(-1)  # [num_tokens, embedding_size / 2]
                positional_embeddings = torch.cat([torch.sin(positions), torch.cos(positions)], dim=-1)

        all_embeddings_mixed = torch.cat((embedded_tokens, positional_embeddings), dim=0)
        return all_embeddings_mixed[base_idxs_of_tokens]


class MixedStrElementRepresentationModel(
    AbstractNeuralModel[str, TensorizedStr, PosEncodingMixedEmbedder],
    AbstractNodeEmbedder,
):
    """
    Combine positional encodings with standard encodings.

    Tokens starting with a given prefix e.g. $posNNN will be transformed to the positional encoding of NNN.
    All other tokens, will be transformed by the base model.
    """

    def __init__(self, base_embedder_model: StrElementRepresentationModel, pos_encoding_prefix: str = "$pos"):
        super().__init__()
        self.base_embedder_model = base_embedder_model
        self.pos_encoding_prefix: Final = pos_encoding_prefix

    def representation_size(self) -> int:
        return self.base_embedder_model.representation_size

    def update_metadata_from(self, datapoint: str) -> None:
        if not datapoint.startswith(self.pos_encoding_prefix):
            self.base_embedder_model.update_metadata_from(datapoint)

    def build_neural_module(self) -> PosEncodingMixedEmbedder:
        return PosEncodingMixedEmbedder(
            embedding_size=self.base_embedder_model.embedding_size,
            base_embedder=self.base_embedder_model.build_neural_module(),
        )

    def tensorize(self, datapoint: str) -> TensorizedStr:
        if datapoint.startswith(self.pos_encoding_prefix):
            return TensorizedStr(int(datapoint[len(self.pos_encoding_prefix) :]), True)
        return TensorizedStr(self.base_embedder_model.tensorize(datapoint), is_positional=False)

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            "base_model_tokens": self.base_embedder_model.initialize_minibatch(),
            "num_base_elements": 0,
            "positional_tokens": [],
            "element_index": [],
            "is_positional": [],
        }

    def extend_minibatch_with(self, tensorized_datapoint: TensorizedStr, partial_minibatch: Dict[str, Any]) -> bool:
        partial_minibatch["is_positional"].append(tensorized_datapoint.is_positional)
        if tensorized_datapoint.is_positional:
            partial_minibatch["element_index"].append(len(partial_minibatch["positional_tokens"]))
            partial_minibatch["positional_tokens"].append(tensorized_datapoint.data)
            continue_extending = True
        else:
            partial_minibatch["element_index"].append(partial_minibatch["num_base_elements"])
            continue_extending = self.base_embedder_model.extend_minibatch_with(
                tensorized_datapoint.data, partial_minibatch["base_model_tokens"]
            )
            partial_minibatch["num_base_elements"] += 1

        return continue_extending

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:

        is_positional = np.array(accumulated_minibatch_data["is_positional"], dtype=np.bool)
        positions = np.array(accumulated_minibatch_data["element_index"], dtype=np.int)

        # Offset the positions of the positional encodings, since they will be concatenated in the nn.Module
        positions += is_positional * accumulated_minibatch_data["num_base_elements"]

        return {
            "base_model_tokens": self.base_embedder_model.finalize_minibatch(
                accumulated_minibatch_data["base_model_tokens"], device=device
            ),
            "positional_tokens": torch.tensor(
                accumulated_minibatch_data["positional_tokens"], device=device, dtype=torch.long
            ),
            "base_idxs_of_tokens": torch.tensor(positions, dtype=torch.int64, device=device),
        }
