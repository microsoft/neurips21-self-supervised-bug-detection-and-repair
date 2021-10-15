import math
from typing import Any, Callable, Dict, Union

import torch
import torch.nn as nn
from ptgnn.baseneuralmodel import ModuleWithMetrics

from buglab.models.utils import scatter_log_softmax, scatter_max


class LocalizationModule(ModuleWithMetrics):
    def __init__(
        self,
        representation_size: int,
        buggy_samples_weight_schedule: Callable[[int], float],
        abstain_weight: float = 0.0,
    ):
        """
        :param buggy_samples_weight: a float or a callable that returns the weight for a given epoch from >=0
        """
        super().__init__()
        self._summary_repr = nn.Linear(in_features=representation_size, out_features=representation_size, bias=True)
        self._l1 = nn.Linear(in_features=2 * representation_size, out_features=representation_size, bias=True)
        self._repr_to_localization_score = nn.Linear(in_features=representation_size, out_features=1, bias=False)

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
            "Localization Accuracy": self.__num_correct / self.__total_num_samples,
            "No Bug Recall": self.__num_no_bug_correct / self.__num_no_bug,
            "Localization Loss": self.__localization_loss / self.__total_num_samples,
            "Weight of Buggy Samples": self._buggy_samples_weight_schedule(self._epoch_idx),
        }

    def compute_localization_logprobs(self, candidate_reprs, candidate_to_sample_idx, num_samples):
        """
        :param candidate_reprs: [num_candidates, H]
        :param candidate_to_sample_idx: [num_candidates]
        """
        sample_summary = scatter_max(self._summary_repr(candidate_reprs), index=candidate_to_sample_idx, dim=0)[0][
            candidate_to_sample_idx
        ]
        l1_out = torch.sigmoid(self._l1(torch.cat([candidate_reprs, sample_summary], dim=-1)))
        candidate_scores = self._repr_to_localization_score(l1_out).squeeze(-1)  # [num_candidates]

        arange = torch.arange(num_samples, dtype=torch.int64, device=candidate_to_sample_idx.device)
        candidate_node_scores_with_no_bug = torch.cat(
            (
                candidate_scores,
                torch.ones_like(arange, dtype=torch.float32),
            )
        )  # [B+C]
        candidate_to_sample_idx = torch.cat(
            (candidate_to_sample_idx, arange)
        ).contiguous()  # [num_samples + num_candidates]
        candidate_node_log_probs = scatter_log_softmax(
            candidate_node_scores_with_no_bug, candidate_to_sample_idx
        )  # [num_samples + num_candidates]

        return candidate_to_sample_idx, candidate_node_log_probs, arange

    def forward(self, candidate_reprs, candidate_to_sample_idx, has_bug, correct_candidate_idxs):
        candidate_to_sample_idx, candidate_log_probs, arange = self.compute_localization_logprobs(
            candidate_reprs, candidate_to_sample_idx, has_bug.shape[0]
        )

        correct_candidate_idxs = torch.where(
            has_bug,
            correct_candidate_idxs,
            arange + candidate_reprs.shape[0],
        )  # [B]

        log_probs_per_sample = candidate_log_probs[correct_candidate_idxs]
        log_probs_per_sample = log_probs_per_sample.clamp(min=-math.inf, max=math.log(0.995))

        if hasattr(self, "_abstain_weight") and self._abstain_weight > 0:
            log_probs_per_sample = log_probs_per_sample + torch.where(
                has_bug,
                self._abstain_weight * candidate_log_probs[arange + candidate_reprs.shape[0]],
                torch.zeros_like(log_probs_per_sample),
            )

        with torch.no_grad():
            self.__total_num_samples += float(log_probs_per_sample.shape[0])
            correct_samples = (
                scatter_max(candidate_log_probs, candidate_to_sample_idx, dim_size=has_bug.shape[0])[1]
                == correct_candidate_idxs
            )
            self.__num_correct += float(correct_samples.sum().cpu())

            self.__num_no_bug += int(has_bug.logical_not().sum().cpu())
            self.__num_no_bug_correct += int((has_bug.logical_not() * correct_samples).sum().cpu())

            self.__localization_loss += float(-log_probs_per_sample.sum().cpu())

        buggy_samples_weight = self._buggy_samples_weight_schedule(self._epoch_idx)
        if buggy_samples_weight == 1.0:
            return -log_probs_per_sample.mean()

        weights = torch.where(
            has_bug,
            torch.full_like(log_probs_per_sample, fill_value=buggy_samples_weight),
            torch.full_like(log_probs_per_sample, fill_value=1.0),
        )
        return -(log_probs_per_sample * weights).sum() / weights.sum()
