import typing as th
import abc
import networkx as nx
import torch
import os
import sys
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_DIR, "../../"))
from oslow.evaluation import backward_relative_penalty, count_backward, shd, sid
from oslow.data import OCDDataset


class AbstractBaseline(abc.ABC):
    """Abstract class for running baselines that estimate causal orderings.

    Given a OCDDataset object (which has a dag and samples attribute), the baseline
    should estimate the causal ordering of the variables in the dataset.
    """

    def __init__(
        self,
        name: th.Optional[str] = None,
        standard: bool = False,
    ):
        self.name = self.__class__.__name__ if name is None else name
        self.standard = standard
        self.dataset = None

    def set_dataset(self, dataset: OCDDataset):
        self.dataset = dataset

    @property
    def true_ordering(self) -> th.List[int]:
        """Return the true ordering of the dataset."""
        if self.dataset is None or not hasattr(self.dataset, "dag"):
            raise ValueError(
                "Dataset must have a dag attribute to return the true ordering."
            )
        return list(nx.topological_sort(self.dataset.dag))

    def get_samples(
        self,
        conversion: th.Literal["tensor", "numpy", "pandas"] = "tensor",
    ):
        if self.dataset is None or not hasattr(self.dataset, "dag"):
            raise ValueError("Dataset is not loaded to get the samples.")

        samples = torch.from_numpy(self.dataset.samples.to_numpy())
        if self.standard:
            samples = (samples - samples.mean(dim=0)) / samples.std(dim=0)
        if conversion == "tensor":
            return samples
        elif conversion == "numpy":
            return samples.numpy()
        elif conversion == "pandas":
            return pd.DataFrame(samples.numpy())
        return samples

    @abc.abstractmethod
    def estimate_order(self, **kwargs) -> th.Union[th.List[int], torch.Tensor]:
        """Fit the baseline on the dataset and return the estimated causal orderings.

        Returns:
            adj_matrix list of the estimated orderings, e.g., [2, 0, 1] means X_2 -> X_0 -> X_1
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def estimate_dag(self, **kwargs) -> th.Union[nx.DiGraph, torch.Tensor]:
        raise NotImplementedError()

    def evaluate(self, structure: th.Optional[bool] = True):
        """Evaluate the baseline on the dataset.
        Args:
            structure: Whether to evaluate the structure of the estimated DAG

        Returns:
            adj_matrix dictionary of evaluation metrics
        """
        estimated_order = self.estimate_order()
        # count the number of backward edges
        backward_count = count_backward(estimated_order, self.dataset.dag)
        # compute the backward relative penalty
        backward_penalty = backward_relative_penalty(estimated_order, self.dataset.dag)
        result = {
            "backward_count": backward_count,
            "backward_relative_penalty": backward_penalty,
            "true_ordering": self.true_ordering,
            "estimated_ordering": estimated_order,
        }
        if structure:
            estimated_dag = self.estimate_dag()
            result["SID"] = sid(self.dataset.dag, estimated_dag)
            result["SHD"] = shd(self.dataset.dag, estimated_dag)

        return result
