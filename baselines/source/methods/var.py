# Codes are adopted from the original implementation
# https://github.com/paulrolland1307/SCORE/tree/5e18c73a467428d51486d2f683349dde2607bfe1
# under the GNU Affero General Public License v3.0
# Copy right belongs to the original author https://github.com/paulrolland1307

from source.base import AbstractBaseline  # also adds ocd to sys.path
from source.utils import full_DAG
import torch
import typing as th
import numpy as np
import networkx as nx
from oslow.post_processing.cam_pruning import cam_pruning


class Var(AbstractBaseline):
    def __init__(
        self,
        dataset: th.Union["OCDDataset", str],  # type: ignore
        dataset_args: th.Optional[th.Dict[str, th.Any]] = None,
        # hyperparameters
        standardize: bool = False,
        verbose: bool = False,
    ):
        # set hyperparameters
        standardize = False
        super().__init__(dataset=dataset, dataset_args=dataset_args, name="Var", standardize=standardize)
        self.verbose = verbose
        self.data = self.get_data(conversion="tensor")

    def estimate_order(self):
        data = self.data
        # compute the variances of each variable
        with torch.no_grad():
            var = data.var(dim=0)
            order = torch.argsort(var, descending=False).tolist()
        self.order = order
        return order

    def estimate_dag(self):
        dag = full_DAG(self.order if hasattr(self, "order") else self.estimate_order())
        dag = cam_pruning(dag, np.array(self.data.detach().cpu().numpy()), cutoff=0.001, verbose=self.verbose)
        return nx.DiGraph(dag)
