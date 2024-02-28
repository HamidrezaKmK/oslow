from source.base import AbstractBaseline
from cdt.causality.graph import CAM as CDT_CAM
import networkx as nx

import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()


class CAM(AbstractBaseline):
    """CAM baseline from CDT"""

    def __init__(
        self,
        standard: bool = False,
        linear: bool = False,
        verbose: bool = False,
    ):
        super().__init__(name="CAM", standard=standard)
        self.linear = linear
        self.verbose = verbose

    def estimate_order(self):
        samples = self.get_samples(conversion="pandas")
        graph = CDT_CAM(
            score="linear" if self.linear else "nonlinear",
            pruning=False,
            njobs=CPU_COUNT - 1,
            verbose=self.verbose,
        ).predict(samples)
        orders = list(nx.topological_sort(graph))
        return orders

    def estimate_dag(self):
        samples = self.get_samples(conversion="pandas")
        graph = CDT_CAM(
            score="linear" if self.linear else "nonlinear",
            pruning=True,
            njobs=CPU_COUNT - 1,
            verbose=self.verbose,
        ).predict(samples)
        return graph
