from typing import Any, Callable, Dict, Literal, NamedTuple

import logging
import torch
from ptgnn.baseneuralmodel import ModuleWithMetrics

from buglab.models.layers.fixermodules import ArgSwapScoringModule, TextRewriteScoringModule, VarSwapScoringModule
from buglab.models.layers.localizationmodule import (
    CandidateAndRewriteQueryPointerNetLocalizationModule,
    CandidateQueryPointerNetLocalizationModule,
    RewriteReprQueryPointerNetLocalizationModule,
)
from buglab.models.utils import scatter_log_softmax, scatter_max, scatter_min, scatter_sum

LOGGER = logging.getLogger(__name__)


class RewriteChooserInformation(NamedTuple):
    """Struct to hold intermediate information for BugLab modules, abstracting away
    from details on how representations are computed and indexed.

    Used shape abbreviations:
        D ~ hidden dimension of core code embedding model
        num_tr ~ number of text rewrite candidates in batch
        num_vs ~ number of var swap candidates in batch
        num_as ~ number of arg swap candidates in batch
        num_cands ~ number of rewrite candidates in batch (= num_tr + num_vs + num_as)
        num_locs ~ number of rewritable locations in batch (< num_cands, as there are sometimes
            several candidates for a single location)
        num_samples ~ number of samples in batch (< num_locs, as each sample may have several
            rewritable locations)

    Attributes:
        num_samples: int - number of samples in the batch
        rewritable_loc_reprs: float [num_locs, D] tensor - representation of rewritable
            locations.
        rewritable_loc_to_sample_id: int [num_cands] tensor - maps concatenation of all potential
            rewrites to their respective sample.

        text_rewrite_loc_reprs: float [num_tr, D] tensor - representations of locations
            at wich text rewrites may be applied (a location may appear several times, once
            per candidate rewrite).
        text_rewrite_replacement_ids: int [num_tr] tensor - IDs in rewrite vocabulary of the candidate rewrites.
        text_rewrite_to_loc_group: int [num_tr] tensor - maps each rewrite to a batch-specific
            index for all rewrites referring to the same location.
        text_rewrite_correct_idxs: optional int [num_correct_tr] tensor - indices into [0, num_tr-1]
            of those text rewrites that were correct.

        varswap_loc_reprs: float [num_vs, D] tensor - representations of locations
            at which variable swaps may be applied (a location may appear several times, once
            per candidate replacement variable).
        varswap_replacement_reprs: float [num_vs, D] tensor - representations of
            variables which may be used as replacements, such that the i-th entry represents
            a var that could replace the one represented by varswap_loc_reprs[i].
        varswap_to_loc_group: int [num_vs] tensor - maps each variable to a batch-specific
            index for all rewrites referring to the same location.

        argswap_loc_reprs: float [num_as, D] tensor - representations of locations
            at which argument swaps may be applied (a location may appear several times, once
            per candidate replacement argument).
        argswap_replacement_reprs: float [num_as, D] tensor - representations of
            arguments which may be used as replacements, such that the i-th entry represents
            an arg that could replace the one represented by argswap_loc_reprs[i].
        argswap_to_loc_group: int [num_as] tensor - maps each argument location to a
            batch-specific index for all rewrites referring to the same location.
    """

    # Localization-specific information:
    num_samples: int
    rewritable_loc_reprs: torch.Tensor
    rewritable_loc_to_sample_id: torch.Tensor

    # Text repair-specific information:
    text_rewrite_loc_reprs: torch.Tensor
    text_rewrite_replacement_ids: torch.Tensor
    text_rewrite_to_loc_group: torch.Tensor
    # VarSwap-specific information:
    varswap_loc_reprs: torch.Tensor
    varswap_replacement_reprs: torch.Tensor
    varswap_to_loc_group: torch.Tensor
    # ArgSwap-specific information:
    argswap_loc_reprs: torch.Tensor
    argswap_replacement_reprs: torch.Tensor
    argswap_to_loc_group: torch.Tensor


class RewriteLogprobs(NamedTuple):
    """Struct to hold the outputs of a BugLab model's .forward().

    Attributes:
        localization_logprobs: float [num_locs + num_samples] tensor - log probabilities of considered
            locations. The final num_samples entries are the virtual NoBug locations.
            RewriteChooserInformation.rewrite_to_sample_id maps the first part of the tensor
            to sample ids.
        text_rewrite_logprobs: float [num_tr] tensor - log probabilities of the text rewrites passed
            in as RewriteChooserInformation.text_rewrite_*.
        varswap_logprobs: float [num_vs] tensor - log probabilities of the variable swaps passed
            in as RewriteChooserInformation.varswap_*.
        argswap_logprobs: float [num_as] tensor - log probabilities of the argument swaps passed
            in as RewriteChooserInformation.argswap_*.
    """

    localization_logprobs: torch.Tensor

    text_rewrite_logprobs: torch.Tensor
    varswap_logprobs: torch.Tensor
    argswap_logprobs: torch.Tensor


class RewriteChooserModule(ModuleWithMetrics):
    def __init__(
        self,
        entity_repr_size: int,
        rewrite_vocabulary_size: int = -1,
        buggy_samples_weight_schedule: Callable[[int], float] = lambda _: 1.0,
        repair_weight_schedule: Callable[[int], float] = lambda _: 1.0,
        generator_loss_type: Literal["norm-kl", "norm-rmse", "classify-max-loss", "expectation"] = "norm-kl",
        localization_module_type: Literal[
            "CandidateQuery", "RewriteQuery", "CandidateAndRewriteQuery"
        ] = "CandidateAndRewriteQuery",
    ):
        super().__init__()
        self._entity_repr_size = entity_repr_size
        self._generator_loss_type = generator_loss_type

        if localization_module_type.lower() == "CandidateQuery".lower():
            self._localization_module = CandidateQueryPointerNetLocalizationModule(
                representation_size=self._entity_repr_size, buggy_samples_weight_schedule=buggy_samples_weight_schedule
            )
        elif localization_module_type.lower() == "RewriteQuery".lower():
            self._localization_module = RewriteReprQueryPointerNetLocalizationModule(
                representation_size=self._entity_repr_size, buggy_samples_weight_schedule=buggy_samples_weight_schedule
            )
        elif localization_module_type.lower() == "CandidateAndRewriteQuery".lower():
            self._localization_module = CandidateAndRewriteQueryPointerNetLocalizationModule(
                representation_size=self._entity_repr_size, buggy_samples_weight_schedule=buggy_samples_weight_schedule
            )
        else:
            raise ValueError

        self._text_repair_module = TextRewriteScoringModule(self._entity_repr_size, rewrite_vocabulary_size)
        self._varswap_module = VarSwapScoringModule(self._entity_repr_size)
        self._argswap_module = ArgSwapScoringModule(self._entity_repr_size)

        self._buggy_samples_weight_schedule = buggy_samples_weight_schedule
        self._repair_weight_schedule = repair_weight_schedule
        self.__num_train_steps = 0

    def _reset_module_metrics(self) -> None:
        if not hasattr(self, "_epoch_idx"):
            self._epoch_idx = 0
        elif self.training and self.__num_batches > 0:
            # Assumes that module metrics are reset once per epoch.
            self._epoch_idx += 1

        self.__loss = 0.0
        self.__text_rewrite_loss = 0.0
        self.__varswap_loss = 0.0
        self.__argswap_loss = 0.0
        self.__repair_loss = 0.0

        self.__total_samples = 0
        self.__text_rewrite_samples = 0
        self.__varswap_samples = 0
        self.__argswap_samples = 0
        self.__num_batches = 0

    def _module_metrics(self) -> Dict[str, Any]:
        metrics = {
            "Weight of Buggy Samples": self._buggy_samples_weight_schedule(self.__num_train_steps),
            "Weight of Repair": self._repair_weight_schedule(self.__num_train_steps),
        }
        if self.__total_samples > 0:
            metrics["Text Rewrite Loss"] = self.__text_rewrite_loss / (self.__text_rewrite_samples + 1e-7)
            metrics["VarSwap Loss"] = self.__varswap_loss / (self.__varswap_samples + 1e-7)
            metrics["ArgSwap Loss"] = self.__argswap_loss / (self.__argswap_samples + 1e-7)
            metrics["Repair Loss"] = self.__repair_loss / self.__total_samples
        if self.__num_batches > 0:
            metrics["Loss"] = self.__loss / self.__num_batches

        return metrics

    def _compute_rewrite_logprobs(self, rc_info: RewriteChooserInformation):
        text_repair_logits = self._text_repair_module.compute_rewrite_logits(
            target_rewrite_node_representations=rc_info.text_rewrite_loc_reprs,
            candidate_rewrites=rc_info.text_rewrite_replacement_ids,
        )

        varswap_logits = self._varswap_module.compute_per_slot_log_probability(
            slot_representations_per_target=rc_info.varswap_loc_reprs,
            target_nodes_representations=rc_info.varswap_replacement_reprs,
        )

        argswap_logits = self._argswap_module.compute_per_pair_logits(
            slot_representations_per_pair=rc_info.argswap_loc_reprs,
            pair_representations=rc_info.argswap_replacement_reprs,
        )

        # Now compute softmaxes over all logits associated with the same location,
        # and then split it up into logprobs for the individual localization modules again.
        all_logits = torch.cat((text_repair_logits, varswap_logits, argswap_logits))
        logit_groups = torch.cat(
            (
                rc_info.text_rewrite_to_loc_group,
                rc_info.varswap_to_loc_group,
                rc_info.argswap_to_loc_group,
            )
        )
        logprobs = scatter_log_softmax(all_logits, index=logit_groups)
        text_repair_logprobs, varswap_logprobs, argswap_logprobs = torch.split(
            logprobs, [text_repair_logits.shape[0], varswap_logits.shape[0], argswap_logits.shape[0]]
        )
        return argswap_logprobs, text_repair_logprobs, varswap_logprobs

    def _compute_per_location_rewrite_representations(self, rc_info: RewriteChooserInformation):
        # First, we compute representations for all potential rewrites:
        text_rewrite_representations = self._text_repair_module.compute_rewrite_representations(
            node_to_rewrite_representations=rc_info.text_rewrite_loc_reprs,
            rewrite_candidate_information=rc_info.text_rewrite_replacement_ids,
        )
        varswap_representations = self._varswap_module.compute_rewrite_representations(
            node_to_rewrite_representations=rc_info.varswap_loc_reprs,
            rewrite_candidate_information=rc_info.varswap_replacement_reprs,
        )
        argswap_representations = self._argswap_module.compute_rewrite_representations(
            node_to_rewrite_representations=rc_info.argswap_loc_reprs,
            rewrite_candidate_information=rc_info.argswap_replacement_reprs,
        )

        # Now compute the max of the representations at a given location:
        all_rewrite_reprs = torch.cat(
            (text_rewrite_representations, varswap_representations, argswap_representations)
        )  # [num_candidates, D]
        all_groups = torch.cat(
            (rc_info.text_rewrite_to_loc_group, rc_info.varswap_to_loc_group, rc_info.argswap_to_loc_group)
        )  # [num_candidate]
        per_group_max = scatter_max(
            src=all_rewrite_reprs, index=all_groups, dim=0, dim_size=rc_info.rewritable_loc_reprs.shape[0]
        )[
            0
        ]  # [num_target_locations, D]

        return per_group_max

    def compute_rewrite_logprobs(self, rc_info: RewriteChooserInformation) -> RewriteLogprobs:
        # Compute a representation of the rewrites at each rewritable location:
        per_location_rewrite_reprs = self._compute_per_location_rewrite_representations(rc_info)

        # Note: the command
        # torch.exp(torch_scatter.scatter_logsumexp(loc_logprobs, torch.cat((candidate_to_sample_idx, arange))))
        # should give a tensor of ones here (total probabilities).
        loc_logprobs = self._localization_module.compute_localization_logprobs(
            candidate_reprs=rc_info.rewritable_loc_reprs,
            candidate_rewrite_reprs=per_location_rewrite_reprs,
            candidate_to_sample_idx=rc_info.rewritable_loc_to_sample_id,
            num_samples=rc_info.num_samples,
        )

        # Get logprobs for repair rewrites conditional on a target location:
        argswap_logprobs, text_repair_logprobs, varswap_logprobs = self._compute_rewrite_logprobs(rc_info)

        return RewriteLogprobs(
            localization_logprobs=loc_logprobs,
            text_rewrite_logprobs=text_repair_logprobs,
            varswap_logprobs=varswap_logprobs,
            argswap_logprobs=argswap_logprobs,
        )

    def compute_detector_loss(
        self,
        rc_info: RewriteChooserInformation,
        rewrite_logprobs: RewriteLogprobs,
        sample_has_bug: torch.Tensor,
        sample_to_correct_loc_idx: torch.Tensor,
        text_rewrite_correct_idxs: torch.Tensor,
        varswap_correct_idxs: torch.Tensor,
        argswap_correct_idxs: torch.Tensor,
    ):
        """
        Arguments:
            rc_info: struct holding the information required to choose rewrites.
            rewrite_logprobs: struct holding information about chosen rewrites.
            sample_has_bug: [num_samples] - boolean tensor indicating if a sample is buggy or not
            sample_to_correct_loc_idx: [num_samples] - int tensor identifying the correct rewrite
                for buggy samples (and we don't care about its value otherwise)
        """
        localization_loss = self._localization_module.compute_loss(
            candidate_log_probs=rewrite_logprobs.localization_logprobs,
            candidate_to_sample_idx=rc_info.rewritable_loc_to_sample_id,
            sample_has_bug=sample_has_bug,
            sample_to_correct_loc_idx=sample_to_correct_loc_idx,
        )

        with torch.no_grad():
            all_logprobs = torch.cat(
                (
                    rewrite_logprobs.text_rewrite_logprobs,
                    rewrite_logprobs.varswap_logprobs,
                    rewrite_logprobs.argswap_logprobs,
                )
            )
            logprob_groups = torch.cat(
                (rc_info.text_rewrite_to_loc_group, rc_info.varswap_to_loc_group, rc_info.argswap_to_loc_group)
            )
            max_logprob_per_group = scatter_max(all_logprobs, logprob_groups)[0]
            max_logprob_per_rewrite = max_logprob_per_group.gather(-1, logprob_groups)

            rewrite_is_selected = all_logprobs == max_logprob_per_rewrite
            text_rewrite_is_selected_fix, varswap_is_selected_fix, argswap_is_selected_fix = torch.split(
                rewrite_is_selected,
                [
                    rewrite_logprobs.text_rewrite_logprobs.shape[0],
                    rewrite_logprobs.varswap_logprobs.shape[0],
                    rewrite_logprobs.argswap_logprobs.shape[0],
                ],
            )

        text_rewrite_loss = self._text_repair_module.compute_loss(
            rewrite_logprobs.text_rewrite_logprobs, text_rewrite_correct_idxs, text_rewrite_is_selected_fix
        ).sum()
        varswap_loss = self._varswap_module.compute_loss(
            rewrite_logprobs.varswap_logprobs, varswap_correct_idxs, varswap_is_selected_fix
        ).sum()
        argswap_loss = self._argswap_module.compute_loss(
            rewrite_logprobs.argswap_logprobs, argswap_correct_idxs, argswap_is_selected_fix
        ).sum()

        buggy_samples_weight = self._buggy_samples_weight_schedule(self.__num_train_steps)
        repair_weight = self._repair_weight_schedule(self.__num_train_steps)
        repair_loss = text_rewrite_loss + varswap_loss + argswap_loss
        repair_loss = repair_loss * buggy_samples_weight

        total_loss = localization_loss + repair_weight * (repair_loss / sample_has_bug.shape[0])

        with torch.no_grad():
            self.__num_batches += 1
            if self.training:
                self.__num_train_steps += 1
            self.__total_samples += float(sample_has_bug.sum().cpu())
            self.__text_rewrite_samples += text_rewrite_correct_idxs.shape[0]
            self.__varswap_samples += varswap_correct_idxs.shape[0]
            self.__argswap_samples += argswap_correct_idxs.shape[0]
            self.__text_rewrite_loss += float(text_rewrite_loss)
            self.__varswap_loss += float(varswap_loss)
            self.__argswap_loss += float(argswap_loss)
            self.__repair_loss += float(repair_loss)
            self.__loss += float(total_loss)

        return total_loss

    def compute_generator_loss(
        self,
        rc_info: RewriteChooserInformation,
        rewrite_logprobs: RewriteLogprobs,
        text_rewrite_joint_idxs: torch.Tensor,
        varswap_joint_idxs: torch.Tensor,
        argswap_joint_idxs: torch.Tensor,
        # Rewrite logprobs
        observed_rewrite_logprobs: torch.Tensor,
        rewrite_to_sample_id: torch.Tensor,
    ):
        loss_type = self._generator_loss_type

        # We'll accumulate joint logprobs for all rewrites in here:
        rewrite_choice_logprobs = torch.zeros_like(observed_rewrite_logprobs)
        rewrite_choice_to_sample_id = torch.cat(
            (
                rewrite_to_sample_id,
                torch.arange(rc_info.num_samples, dtype=torch.int64, device=rewrite_to_sample_id.device),
            )
        )

        # First we deal with the NO_BUG case.
        no_bug_logprobs = rewrite_logprobs.localization_logprobs[-rc_info.num_samples :]
        rewrite_choice_logprobs[-rc_info.num_samples :] = no_bug_logprobs

        # Text rewrite: combine the location logprobs with the corresponding rewrite logprobs:
        text_rewrite_loc_logprobs = rewrite_logprobs.localization_logprobs[rc_info.text_rewrite_to_loc_group]
        text_rewrite_joint_logprobs = text_rewrite_loc_logprobs + rewrite_logprobs.text_rewrite_logprobs
        rewrite_choice_logprobs[text_rewrite_joint_idxs] = text_rewrite_joint_logprobs

        # Var swap: combine the location logprobs with the corresponding rewrite logprobs:
        varswap_loc_logprobs = rewrite_logprobs.localization_logprobs[rc_info.varswap_to_loc_group]
        varswap_joint_logprobs = varswap_loc_logprobs + rewrite_logprobs.varswap_logprobs
        rewrite_choice_logprobs[varswap_joint_idxs] = varswap_joint_logprobs

        # Arg swap: combine the location logprobs with the corresponding rewrite logprobs:
        argswap_loc_logprobs = rewrite_logprobs.localization_logprobs[rc_info.argswap_to_loc_group]
        argswap_joint_logprobs = argswap_loc_logprobs + rewrite_logprobs.argswap_logprobs
        rewrite_choice_logprobs[argswap_joint_idxs] = argswap_joint_logprobs

        # Now all elements in the following tensor should be very close to 0 (ie. prob=1):
        # scatter_logsumexp(rewrite_choice_logprobs, index=rewrite_choice_to_sample_id)

        observed_rewrites = torch.isinf(observed_rewrite_logprobs).logical_not()
        observed_rewrite_choice_to_sample_id = rewrite_choice_to_sample_id.masked_select(mask=observed_rewrites)
        observed_detection_logprobs = observed_rewrite_logprobs.masked_select(mask=observed_rewrites)
        observed_generation_logprobs = rewrite_choice_logprobs.masked_select(mask=observed_rewrites)

        if loss_type in ("norm-kl", "norm-rmse", "classify-max-loss"):
            # Score matching: the re-normalized detection probs should match the (1-generate_prob)
            observed_generation_logprobs = scatter_log_softmax(
                observed_generation_logprobs,
                index=observed_rewrite_choice_to_sample_id,
            )
            if loss_type == "norm-rmse":
                renormalized_detection_logprobs = scatter_log_softmax(
                    observed_detection_logprobs, index=observed_rewrite_choice_to_sample_id
                )

                # Re-normed probs should sum up to log(1)
                log_difference = torch.logaddexp(renormalized_detection_logprobs, observed_generation_logprobs)
                loss = (log_difference ** 2).mean()
            elif loss_type == "norm-kl":
                failed_detection_logprobs = torch.log(
                    torch.max(
                        1.0 - observed_detection_logprobs.exp(),
                        torch.full_like(observed_detection_logprobs, 1e-30),
                    )
                )
                renormalized_failed_detection_logprobs = scatter_log_softmax(
                    failed_detection_logprobs, index=observed_rewrite_choice_to_sample_id
                )

                # KL among the observed probabilities (re-normalized)
                kl_terms = failed_detection_logprobs.exp() * (
                    renormalized_failed_detection_logprobs - observed_generation_logprobs
                )
                loss = scatter_sum(kl_terms, index=observed_rewrite_choice_to_sample_id).mean()
            elif loss_type == "classify-max-loss":
                _, min_detection_prob_idxs = scatter_min(
                    observed_detection_logprobs, index=observed_rewrite_choice_to_sample_id
                )
                loss = -observed_generation_logprobs[min_detection_prob_idxs].mean()
            else:
                raise ValueError(f"Unknown loss type `{loss_type}`")
        elif loss_type == "expectation":
            # Maximize the expected loss of the discriminator.
            scores = observed_generation_logprobs.exp() * observed_detection_logprobs
            loss = scatter_sum(scores, index=observed_rewrite_choice_to_sample_id).mean()
        else:
            raise ValueError(f"Unknown loss type `{loss_type}`")

        with torch.no_grad():
            self.__loss += float(loss)
            self.__num_batches += 1

        return loss

    def compute_loss(
        self,
        *,
        rc_info: RewriteChooserInformation,
        # Labels for localization:
        sample_has_bug: torch.Tensor,
        sample_to_correct_loc_idx: torch.Tensor,
        # Labels for text repair:
        text_rewrite_correct_idxs: torch.Tensor,
        text_rewrite_joint_idxs: torch.Tensor,
        # Labels for VarSwap:
        varswap_correct_idxs: torch.Tensor,
        varswap_joint_idxs: torch.Tensor,
        # Labels for ArgSwap:
        argswap_correct_idxs: torch.Tensor,
        argswap_joint_idxs: torch.Tensor,
        # Rewrite logprobs
        rewrite_to_sample_id: torch.Tensor,
        # Ignore other args:
        **kwargs,
    ):
        rewrite_logprobs = self.compute_rewrite_logprobs(rc_info)

        # Localization loss changes dependent on if we are in selector/detector setting.
        # We know we are a selector if we have observed_rewrite_logprobs, which is the output
        # of the detector
        observed_rewrite_logprobs = kwargs.get("observed_rewrite_logprobs", None)
        if observed_rewrite_logprobs is not None:
            return self.compute_generator_loss(
                rc_info,
                rewrite_logprobs,
                text_rewrite_joint_idxs=text_rewrite_joint_idxs,
                varswap_joint_idxs=varswap_joint_idxs,
                argswap_joint_idxs=argswap_joint_idxs,
                observed_rewrite_logprobs=observed_rewrite_logprobs,
                rewrite_to_sample_id=rewrite_to_sample_id,
            )
        else:
            return self.compute_detector_loss(
                rc_info,
                rewrite_logprobs,
                sample_has_bug=sample_has_bug,
                sample_to_correct_loc_idx=sample_to_correct_loc_idx,
                text_rewrite_correct_idxs=text_rewrite_correct_idxs,
                varswap_correct_idxs=varswap_correct_idxs,
                argswap_correct_idxs=argswap_correct_idxs,
            )
