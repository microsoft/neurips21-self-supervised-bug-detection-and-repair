import hashlib
from typing import Callable, Iterator, Optional

import torch
import torch_scatter
from dpu_utils.utils import RichPath
from ptgnn.baseneuralmodel import AbstractScheduler
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LambdaLR

from buglab.representations.data import TypeAnnotationData
from buglab.utils.msgpackutils import load_all_msgpack_l_gz


def scatter_log_softmax(src: torch.Tensor, index: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError("`scatter_log_softmax` can only be computed over tensors with floating point data types.")

    with autocast(enabled=False):
        max_value_per_index = torch_scatter.scatter_max(src.float(), index, dim=dim)[0]
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element

    sum_per_index = torch.zeros_like(max_value_per_index).scatter_add_(dim=-1, index=index, src=recentered_scores.exp())
    normalizing_constants = sum_per_index.add_(eps).log_().gather(dim, index)

    return recentered_scores.sub_(normalizing_constants)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: Optional[int] = None):
    return torch_scatter.scatter_sum(src, index, dim, dim_size=dim_size)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: Optional[int] = None):
    with autocast(enabled=False):
        # This can be two scatter_sum ops
        return torch_scatter.scatter_mean(src.float(), index, dim, dim_size=dim_size)


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: Optional[int] = None):
    with autocast(enabled=False):
        return torch_scatter.scatter_max(src.float(), index, dim, dim_size=dim_size)


def scatter_min(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: Optional[int] = None):
    with autocast(enabled=False):
        return torch_scatter.scatter_min(src.float(), index, dim, dim_size=dim_size)


def optimizer(p, lr: float = 0.0001) -> torch.optim.Adam:
    return torch.optim.Adam(p, lr=lr)


class LinearWarmupScheduler(AbstractScheduler):
    def __init__(self, optimizer, num_warmup_steps: int = 800, last_epoch=-1):
        self.__num_warmup_steps = num_warmup_steps
        self.__scheduler = LambdaLR(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, current_step: int):
        if current_step < self.__num_warmup_steps:
            return float(current_step) / float(max(1.0, self.__num_warmup_steps))
        return 1.0

    def step(self, epoch_idx: int, epoch_step: int) -> None:
        self.__scheduler.step()  # Run per-step (not per-epoch)


def construct_type_training_data_callable(
    data_path: RichPath,
    training: bool,
    max_files_per_fold: Optional[int] = None,
    limit_num_yielded_elements: Optional[int] = None,
) -> Callable[[], Iterator[TypeAnnotationData]]:
    def data_it() -> Iterator[TypeAnnotationData]:
        num_yielded = 0
        for datum in load_all_msgpack_l_gz(
            data_path,
            shuffle=training,
            take_only_first_n_files=max_files_per_fold,
            limit_num_yielded_elements=limit_num_yielded_elements,
        ):
            datum: TypeAnnotationData

            name = datum["package_name"] + datum["graph"]["path"]
            hashed = int(hashlib.md5(name.encode()).hexdigest(), 16) % (2 ** 16)
            train_bound = int(2 ** 16 * 0.7)
            if training and hashed < train_bound:
                yield datum
                num_yielded += 1
            elif hashed >= train_bound and not training:
                yield datum
                num_yielded += 1

            if limit_num_yielded_elements is not None and num_yielded > limit_num_yielded_elements:
                return

    return data_it


def compute_generator_loss(
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
):
    # Accumulate logprobs here:
    generation_logprobs = torch.zeros_like(rewrite_logprobs)
    # First we deal with the NO_BUG case.
    num_graphs = arrange.shape[0]
    no_bug_logprobs = localization_logprobs[-num_graphs:]
    generation_logprobs[-num_graphs:] = no_bug_logprobs
    # Text rewrites:
    text_localization_logprobs = localization_logprobs[rewrite_to_location_group]
    unconditional_text_repair_logprobs = text_localization_logprobs + text_repair_logprobs
    generation_logprobs[text_rewrite_idxs] += unconditional_text_repair_logprobs
    # Var misuse:
    var_misuse_localization_logprobs = localization_logprobs[candidate_symbol_to_location_group]
    unconditional_var_misuse_logprobs = var_misuse_localization_logprobs + varmisuse_logprobs
    generation_logprobs[candidate_rewrite_idxs] += unconditional_var_misuse_logprobs
    # Arg swap
    arg_swap_localization_logprobs = localization_logprobs[swapped_pair_to_call_location_group]
    unconditional_arg_swap_logprobs = arg_swap_localization_logprobs + arg_swap_logprobs
    generation_logprobs[pair_rewrite_idxs] += unconditional_arg_swap_logprobs
    # All elements in the following tensor should be very close to 0 (ie. prob=1)
    # scatter_logsumexp(generation_logprobs, index=torch.cat((rewrite_to_graph_id, arrange)))
    observed_rewrites = torch.isinf(rewrite_logprobs).logical_not()
    index = torch.cat((rewrite_to_graph_id, arrange)).masked_select(mask=observed_rewrites)
    observed_detection_logprobs = rewrite_logprobs.masked_select(mask=observed_rewrites)
    observed_generation_logprobs = generation_logprobs.masked_select(mask=observed_rewrites)
    if loss_type in ("norm-kl", "norm-rmse", "classify-max-loss"):
        # Score matching: the re-normalized detection probs should match the (1-generate_prob)

        observed_generation_logprobs = scatter_log_softmax(
            observed_generation_logprobs,
            index=index,
        )
        if loss_type == "norm-rmse":
            renormalized_detection_logprobs = scatter_log_softmax(observed_detection_logprobs, index=index)

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
            renormalized_failed_detection_logprobs = scatter_log_softmax(failed_detection_logprobs, index=index)

            # KL among the observed probabilities (re-normalized)
            kl_terms = failed_detection_logprobs.exp() * (
                renormalized_failed_detection_logprobs - observed_generation_logprobs
            )
            loss = scatter_sum(kl_terms, index=index).mean()
        elif loss_type == "classify-max-loss":
            _, min_detection_prob_idxs = scatter_min(observed_detection_logprobs, index=index)
            loss = -observed_generation_logprobs[min_detection_prob_idxs].mean()
        else:
            raise ValueError(f"Unknown loss type `{loss_type}`")
    elif loss_type == "expectation":
        # Maximize the expected loss of the discriminator.
        scores = observed_generation_logprobs.exp() * observed_detection_logprobs
        loss = scatter_sum(scores, index=index).mean()
    else:
        raise ValueError(f"Unknown loss type `{loss_type}`")
    return loss
