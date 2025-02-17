from abc import ABC, abstractmethod
from typing import List

import torch


class MultiTaskAnnealingStrategy(ABC):
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks

    @abstractmethod
    def __call__(self, iter: int, *args, **kwargs) -> List[float]:
        pass


class DWAMultiTaskAnnealing(MultiTaskAnnealingStrategy):
    """
    Dynamic Weight Average (DWA) from https://arxiv.org/pdf/1803.10704
    """

    MIN_LAMBDA_VAL = 0.1

    def __init__(self, num_tasks: int, temperature: float):
        super().__init__(num_tasks=num_tasks)
        self.temperature = temperature
        self.loss_t_minus_1 = None
        self.loss_t_minus_2 = None

    def __call__(self, iter: int, losses: torch.Tensor | None = None):
        if losses is None:
            return [1.0 for _ in range(self.num_tasks)]

        if self.loss_t_minus_1 is None:
            self.loss_t_minus_1 = losses.detach()

        if self.loss_t_minus_2 is None:
            self.loss_t_minus_2 = losses.detach()

        weights = self.loss_t_minus_1 / self.loss_t_minus_2
        self.loss_t_minus_2 = self.loss_t_minus_1
        self.loss_t_minus_1 = losses.detach()
        exp_weights_temp = torch.exp(weights / self.temperature)
        lambdas = exp_weights_temp / exp_weights_temp.sum() * self.num_tasks

        if torch.isnan(lambdas).any():
            # Ensure that the minimum lambda value is `MIN_LAMBDA_VAL`
            lambdas = torch.nan_to_num(lambdas, self.num_tasks)
            lambdas = torch.max(lambdas, torch.tensor([self.MIN_LAMBDA_VAL]))

            lambdas /= lambdas.sum()

        return lambdas.tolist()
