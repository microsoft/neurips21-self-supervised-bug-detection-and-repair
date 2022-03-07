from typing import Optional

import torch
import torch_scatter
from ptgnn.baseneuralmodel import AbstractScheduler
from torch.optim.lr_scheduler import LambdaLR
from torch_scatter import scatter_max, scatter_sum


def scatter_log_softmax(
    src: torch.Tensor, index: torch.Tensor, dim: int = -1, eps: float = 1e-12, dim_size: Optional[int] = None
) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError("`scatter_log_softmax` can only be computed over tensors with floating point data types.")

    max_value_per_index = torch_scatter.scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element

    sum_per_index = torch_scatter.scatter_sum(src=recentered_scores.exp(), index=index, dim=-1, dim_size=dim_size)
    normalizing_constants = sum_per_index.log_().gather(dim, index)

    return recentered_scores.sub_(normalizing_constants)


class AdamOptimizerWithProfiler(torch.optim.Adam):
    def __init__(self, params, profiler=None, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.profiler = profiler

    def step(self, closure=None):
        try:
            return super().step(closure)
        finally:
            self.profiler.step()


def optimizer(p, lr: float = 0.0001, profiler=None) -> torch.optim.Adam:
    if profiler is None:
        return torch.optim.Adam(p, lr=lr)
    return AdamOptimizerWithProfiler(p, profiler=profiler, lr=lr)


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
