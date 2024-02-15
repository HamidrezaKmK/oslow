from source.base import AbstractBaseline
from cdt.causality.graph import CAM as CDT_CAM
import networkx as nx
import typing as th

import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()


class CAM(AbstractBaseline):
    """CAM baseline from CDT"""

    def __init__(
        self,
        dataset: th.Union["OCDDataset", str],  # type: ignore
        dataset_args: th.Optional[th.Dict[str, th.Any]] = None,
        # hyperparameters
        standardize: bool = False,
        linear: bool = False,
        verbose: bool = False,
    ):
        super().__init__(dataset=dataset, dataset_args=dataset_args, name="CAM", standardize=standardize)
        self.linear = linear
        self.verbose = verbose
        if self.linear:
            raise NotImplementedError("Linear CAM does not work!")

    def estimate_order(self):
        samples = self.get_data(conversion="pandas")
        graph = CDT_CAM(
            score="linear" if self.linear else "nonlinear", pruning=False, njobs=CPU_COUNT - 1, verbose=self.verbose
        ).predict(samples)
        orders = list(nx.topological_sort(graph))
        return orders

    def estimate_dag(self):
        samples = self.get_data(conversion="pandas")
        graph = CDT_CAM(
            score="linear" if self.linear else "nonlinear", pruning=True, njobs=CPU_COUNT - 1, verbose=self.verbose
        ).predict(samples)
        return graph
