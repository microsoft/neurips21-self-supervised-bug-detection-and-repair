from typing import Any, Callable, Dict

import math
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from ptgnn.baseneuralmodel import ModuleWithMetrics

from buglab.models.layers.mlp import MLP
from buglab.models.utils import scatter_log_softmax, scatter_max


class LocalizationModule(ModuleWithMetrics, ABC):
    def __init__(
        self,
        *,
        representation_size: int,
        buggy_samples_weight_schedule: Callable[[int], float],
        abstain_weight: float = 0.0,
    ):
        """
        :param buggy_samples_weight: a float or a callable that returns the weight for a given epoch from >=0
        """
        super().__init__()
        self._representation_size = representation_size
        self._buggy_samples_weight_schedule = buggy_samples_weight_schedule
        self._abstain_weight = abstain_weight

    def _reset_module_metrics(self) -> None:
        if not hasattr(self, "_epoch_idx"):
            self._epoch_idx = 0
        elif self.training and self.__total_num_samples > 0:
            # Assumes that module metrics are reset once per epoch.
            self._epoch_idx += 1

        self.__num_correct = 0
        self.__localization_loss = 0.0
        self.__num_no_bug = 0
        self.__num_no_bug_correct = 0

        self.__total_num_samples = 0

    def _module_metrics(self) -> Dict[str, Any]:
        if self.__total_num_samples == 0:
            return {}
        return {
            "Localization Accuracy": int(self.__num_correct) / int(self.__total_num_samples),
            "No Bug Recall": int(self.__num_no_bug_correct) / int(self.__num_no_bug),
            "Localization Loss": float(self.__localization_loss) / int(self.__total_num_samples),
        }

    @abstractmethod
    def compute_localization_logprobs(
        self, candidate_reprs, candidate_rewrite_reprs, candidate_to_sample_idx, num_samples
    ):
        """
        :param candidate_reprs: [num_candidates, H] - Representation of each rewritable location.
        :param candidate_rewrite_reprs: [num_candidates, H] - Representation of the rewrites at
            each rewritable location.
        :param candidate_to_sample_idx: [num_candidates] - Index tensor, mapping each candidate
            to a sample. (Needed to do a softmax)
        :param num_samples: int scalar - number of samples in our input
        """
        raise NotImplementedError()

    def compute_loss(
        self,
        candidate_log_probs: torch.FloatTensor,  # [num_candidates + num_samples]
        candidate_to_sample_idx: torch.IntTensor,  # [num_candidates]
        sample_has_bug: torch.BoolTensor,  # [num_samples]
        sample_to_correct_loc_idx: torch.IntTensor,  # [num_samples]
    ) -> torch.FloatTensor:
        # candidate_log_probs is made of two parts - actual candidates, and then "virtual" NoBug locations.
        # We need to treat these differently from time to time...
        num_candidates = candidate_to_sample_idx.shape[0]
        num_samples = sample_has_bug.shape[0]
        sample_range = torch.arange(num_samples, dtype=torch.int64, device=sample_to_correct_loc_idx.device)
        no_bug_indices = num_candidates + sample_range
        candidate_to_sample_idx = torch.cat((candidate_to_sample_idx, sample_range))

        # The input sample_to_correct_loc_idx has garbage for samples that are not buggy; replace those values:
        sample_to_correct_loc_idx = torch.where(sample_has_bug, sample_to_correct_loc_idx, no_bug_indices)  # [B]

        log_probs_per_sample = candidate_log_probs[sample_to_correct_loc_idx]
        log_probs_per_sample = log_probs_per_sample.clamp(min=-math.inf, max=math.log(0.995))
        if hasattr(self, "_abstain_weight") and self._abstain_weight > 0:
            log_probs_per_sample = log_probs_per_sample + torch.where(
                sample_has_bug,
                self._abstain_weight * candidate_log_probs[no_bug_indices],
                torch.zeros_like(log_probs_per_sample),
            )

        with torch.no_grad():
            self.__total_num_samples += num_samples
            correct_samples = (
                scatter_max(candidate_log_probs, candidate_to_sample_idx, dim_size=num_samples)[1]
                == sample_to_correct_loc_idx
            )
            self.__num_correct += correct_samples.sum()

            self.__num_no_bug += sample_has_bug.logical_not().sum()
            self.__num_no_bug_correct += (sample_has_bug.logical_not() * correct_samples).sum()

            self.__localization_loss += -log_probs_per_sample.sum()

        buggy_samples_weight = self._buggy_samples_weight_schedule(self._epoch_idx)
        if buggy_samples_weight == 1.0:
            return -log_probs_per_sample.mean()

        weights = torch.where(
            sample_has_bug,
            torch.full_like(log_probs_per_sample, fill_value=buggy_samples_weight),
            torch.full_like(log_probs_per_sample, fill_value=1.0),
        )
        return -(log_probs_per_sample * weights).sum() / weights.sum()

    def forward(
        self, candidate_reprs, candidate_rewrite_reprs, candidate_to_sample_idx, sample_has_bug, correct_candidate_idxs
    ):
        """
        :param candidate_reprs: [num_candidates, H] - Representation of each rewritable location.
        :param candidate_rewrite_reprs: [num_candidates, H] - Representation of the rewrites at
            each rewritable location.
        :param candidate_to_sample_idx: [num_candidates] - Index tensor, mapping each candidate
            to a sample. (Needed to do a softmax)
        :param sample_has_bug: [num_samples] - boolean tensor indicating if a sample is buggy or not
        :param correct_candidate_idxs: [num_samples] - int tensor identifying the correct rewrite
            for buggy samples (and we don't care about its value otherwise)
        """
        num_samples = sample_has_bug.shape[0]
        candidate_log_probs = self.compute_localization_logprobs(
            candidate_reprs, candidate_rewrite_reprs, candidate_to_sample_idx, num_samples
        )

        return self.compute_loss(candidate_log_probs, candidate_to_sample_idx, sample_has_bug, correct_candidate_idxs)


class RewriteReprQueryPointerNetLocalizationModule(LocalizationModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scoring_mlp = MLP(
            input_dim=2 * self._representation_size, out_dim=1, hidden_layer_dims=[self._representation_size]
        )
        self._nobug_score = nn.Parameter(torch.zeros(1))

    def compute_localization_logprobs(
        self, candidate_reprs, candidate_rewrite_reprs, candidate_to_sample_idx, num_samples
    ):
        """See parent class."""
        candidate_scores = self._scoring_mlp(torch.cat((candidate_reprs, candidate_rewrite_reprs), dim=-1)).squeeze(-1)
        arange = torch.arange(num_samples, dtype=torch.int64, device=candidate_to_sample_idx.device)
        candidate_to_sample_idx = torch.cat(
            (candidate_to_sample_idx, arange)
        ).contiguous()  # [num_samples + num_candidates]
        candidate_log_probs = scatter_log_softmax(
            src=torch.cat([candidate_scores, self._nobug_score.repeat(num_samples)]),
            index=candidate_to_sample_idx,
        )  # [num_candidates + num_samples]

        return candidate_log_probs


class CandidateQueryPointerNetLocalizationModule(LocalizationModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._summary_repr = nn.Linear(
            in_features=self._representation_size, out_features=self._representation_size, bias=True
        )
        self._l1 = nn.Linear(
            in_features=2 * self._representation_size, out_features=self._representation_size, bias=True
        )
        self._repr_to_localization_score = nn.Linear(in_features=self._representation_size, out_features=1, bias=False)

    def compute_localization_logprobs(
        self, candidate_reprs, candidate_rewrite_reprs, candidate_to_sample_idx, num_samples
    ):
        """See parent class."""
        sample_summary = scatter_max(
            self._summary_repr(candidate_reprs), index=candidate_to_sample_idx, dim=0, dim_size=num_samples
        )[0][
            candidate_to_sample_idx
        ]  # [num_candidates, D] - for each cand, the element-wise max across all candidates for the same location
        l1_out = torch.sigmoid(self._l1(torch.cat([candidate_reprs, sample_summary], dim=-1)))
        candidate_scores = self._repr_to_localization_score(l1_out).squeeze(-1)  # [num_candidates]

        arange = torch.arange(num_samples, dtype=torch.int64, device=candidate_to_sample_idx.device)
        candidate_scores_with_no_bug = torch.cat(
            (
                candidate_scores,
                torch.ones_like(arange, dtype=torch.float),
            )
        )  # [B+C]
        candidate_to_sample_idx = torch.cat(
            (candidate_to_sample_idx, arange)
        )  # .contiguous()  # [num_candidates + num_samples]
        candidate_log_probs = scatter_log_softmax(
            candidate_scores_with_no_bug, candidate_to_sample_idx
        )  # [num_candidates + num_samples]

        return candidate_log_probs


class CandidateAndRewriteQueryPointerNetLocalizationModule(LocalizationModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._summary_mlp = MLP(
            input_dim=2 * self._representation_size,
            out_dim=self._representation_size,
            hidden_layer_dims=[self._representation_size] * 2,
        )
        self._scoring_mlp = MLP(
            input_dim=3 * self._representation_size, out_dim=1, hidden_layer_dims=[self._representation_size] * 3
        )
        self._nobug_score = nn.Parameter(torch.zeros(1))

    def compute_localization_logprobs(
        self, candidate_reprs, candidate_rewrite_reprs, candidate_to_sample_idx, num_samples
    ):
        """See parent class."""
        # For each sample, the max over a projection of all candidate locations in the sample:
        candidate_summaries = self._summary_mlp(torch.cat((candidate_reprs, candidate_rewrite_reprs), dim=-1))
        sample_summaries = scatter_max(candidate_summaries, index=candidate_to_sample_idx, dim=0, dim_size=num_samples)[
            0
        ]
        per_candidate_sample_summaries = sample_summaries[candidate_to_sample_idx]  # [num_candidates, D]

        # Compute per-candidate scores from the location representation, the rewrite options, and the
        # pooled info about the sample:
        candidate_scores = self._scoring_mlp(
            torch.cat((candidate_reprs, candidate_rewrite_reprs, per_candidate_sample_summaries), dim=-1)
        )  # [num_candidates + num_samples, 1]

        # Finally, compute logprobs from the scores and the (learnable) NoBug score:
        candidate_logprobs = scatter_log_softmax(
            src=torch.cat((candidate_scores.squeeze(-1), self._nobug_score.expand(num_samples))),
            index=torch.cat(
                (
                    candidate_to_sample_idx,
                    torch.arange(num_samples, dtype=torch.int64, device=candidate_to_sample_idx.device),
                )
            ),
            dim_size=num_samples,
        )  # [num_candidates + num_samples]

        return candidate_logprobs
