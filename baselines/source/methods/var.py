from source.base import AbstractBaseline  # also adds ocd to sys.path
from source.utils import full_DAG
import torch
import numpy as np
import networkx as nx
from oslow.post_processing.cam_pruning import cam_pruning


class Var(AbstractBaseline):
    def __init__(
        self,
        standard: bool = False,
        verbose: bool = False,
    ):
        # set hyperparameters
        super().__init__(name="VarSort", standard=standard)
        self.verbose = verbose

    def estimate_order(self):
        data = self.get_samples(conversion="tensor")
        # compute the variances of each variable
        with torch.no_grad():
            var = data.var(dim=0)
            order = torch.argsort(var, descending=False).tolist()
        self.order = order
        return order

    def estimate_dag(self):
        dag = full_DAG(self.order if hasattr(self, "order") else self.estimate_order())
        dag = cam_pruning(
            dag,
            np.array(self.data.detach().cpu().numpy()),
            cutoff=0.001,
            verbose=self.verbose,
        )
        return nx.DiGraph(dag)
