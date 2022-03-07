from typing import Any, Dict

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from ptgnn.baseneuralmodel import ModuleWithMetrics

from buglab.models.layers.mlp import MLP


class RewriteScoringModule(ModuleWithMetrics, ABC):
    def __init__(self, rewrite_name: str, input_representation_size: int):
        super().__init__()
        self._name = rewrite_name
        self._input_dim = input_representation_size

    def _reset_module_metrics(self) -> None:
        self._num_correct = 0
        self._num_samples = 0

    def _module_metrics(self) -> Dict[str, Any]:
        if self._num_samples == 0:
            return {}
        return {
            f"{self._name} Fixer Accuracy": int(self._num_correct) / int(self._num_samples),
            f"{self._name} Fixer Stats": f"{int(self._num_correct) / int(self._num_samples):.2%} ({int(self._num_correct)}/{int(self._num_samples)})",
        }

    def compute_loss(self, candidate_logprobs, correct_candidate_idx, selected_fixes=None) -> torch.Tensor:
        """
        :param candidate_logprobs: float [N] tensor, given log-probabilities for each candidate
        :param correct_candidate_idx: long [B] tensor, with values in [0..N-1] identifying correct choices.
        :param selected_fixes: optional bool [N] tensor, indicating if a candidate had maximal logprob for the loc
        """
        if selected_fixes is not None:
            with torch.no_grad():
                self._num_correct += selected_fixes[correct_candidate_idx].sum()
                self._num_samples += correct_candidate_idx.shape[0]
        return -candidate_logprobs[correct_candidate_idx]

    @abstractmethod
    def compute_rewrite_representations(self, node_to_rewrite_representations, rewrite_candidate_information):
        """
        Compute a representation of each rewrite, useful for scoring locations.

        :param node_to_rewrite_representations: [N, D] - representation of a location that can be
            rewritten.
        :param rewrite_candidate_information: [N, ...] - information about the rewrite at each location;
            repair module-specific shape.
        :return: [N, D] - representation of each location/rewrite pair.
        """
        pass


class TextRewriteScoringModule(RewriteScoringModule):
    def __init__(self, input_representation_size: int, rewrite_vocab_size: int):
        super().__init__(input_representation_size=input_representation_size, rewrite_name="Text Rewrite")
        self.__text_rewrite_embeddings = nn.Embedding(rewrite_vocab_size, embedding_dim=input_representation_size)
        self.__text_rewrite_scorer = MLP(
            input_dim=2 * input_representation_size,
            out_dim=1,
            hidden_layer_dims=[input_representation_size],
        )

        self.__to_rewrite_repr_mlp = MLP(
            input_dim=2 * input_representation_size,
            out_dim=input_representation_size,
            hidden_layer_dims=[input_representation_size],
        )

    def compute_rewrite_logits(self, target_rewrite_node_representations, candidate_rewrites):
        """
        :param target_rewrite_node_representations: [N, D]
        :param candidate_rewrites: [N]
        """
        embedded_target_rewrites = self.__text_rewrite_embeddings(candidate_rewrites)  # [N, D]
        return self.__text_rewrite_scorer(
            torch.cat((embedded_target_rewrites, target_rewrite_node_representations), dim=-1)
        ).squeeze(-1)

    def compute_rewrite_representations(self, node_to_rewrite_representations, rewrite_candidate_information):
        """See super class."""
        return self.__to_rewrite_repr_mlp(
            torch.cat(
                (node_to_rewrite_representations, self.__text_rewrite_embeddings(rewrite_candidate_information)), dim=-1
            )
        )


class VarSwapScoringModule(RewriteScoringModule):
    def __init__(self, input_representation_size: int):
        super().__init__(input_representation_size=input_representation_size, rewrite_name="VarSwap")
        self.__candidate_scorer = MLP(
            input_dim=2 * input_representation_size,
            out_dim=1,
            hidden_layer_dims=[input_representation_size],
        )

        self.__to_rewrite_repr_mlp = MLP(
            input_dim=2 * input_representation_size,
            out_dim=input_representation_size,
            hidden_layer_dims=[input_representation_size],
        )

    def compute_per_slot_log_probability(self, slot_representations_per_target, target_nodes_representations):
        """
        :param slot_representations_per_target:  [N, D]
        :param target_nodes_representations: [N, D]
        """
        # Compute candidate score by applying MLP to combination of slot and candidate representations:
        return self.__candidate_scorer(
            torch.cat((slot_representations_per_target, target_nodes_representations), dim=-1)
        ).squeeze(-1)

    def compute_rewrite_representations(self, node_to_rewrite_representations, rewrite_candidate_information):
        """See super class."""
        return self.__to_rewrite_repr_mlp(
            torch.cat((node_to_rewrite_representations, rewrite_candidate_information), dim=-1)
        )


class ArgSwapScoringModule(RewriteScoringModule):
    def __init__(self, input_representation_size: int):
        super().__init__(input_representation_size=input_representation_size, rewrite_name="ArgSwap")
        self.__pair_scorer = MLP(
            input_dim=3 * input_representation_size,
            out_dim=1,
            hidden_layer_dims=[input_representation_size],
        )

        self.__to_rewrite_repr_mlp = MLP(
            input_dim=3 * input_representation_size,
            out_dim=input_representation_size,
            hidden_layer_dims=[input_representation_size],
        )

    def compute_per_pair_logits(self, slot_representations_per_pair, pair_representations):
        """
        :param slot_representations_per_pair: [N, D]
        :param pair_representations: [N, 2, D]
        """
        # Compute candidate score by applying MLP to combination of slot and pairs representations:
        return self.__pair_scorer(
            torch.cat(
                (
                    slot_representations_per_pair,
                    pair_representations.reshape(pair_representations.shape[0], 2 * self._input_dim),
                ),
                dim=-1,
            )
        ).squeeze(-1)

    def compute_rewrite_representations(self, node_to_rewrite_representations, rewrite_candidate_information):
        """See super class."""
        pair_representations_flat = rewrite_candidate_information.reshape(
            rewrite_candidate_information.shape[0], 2 * self._input_dim
        )
        return self.__to_rewrite_repr_mlp(
            torch.cat((node_to_rewrite_representations, pair_representations_flat), dim=-1)
        )
