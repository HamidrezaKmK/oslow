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
from source.methods.score.stein import SCORE


class Score(AbstractBaseline):
    def __init__(
        self,
        dataset: th.Union["OCDDataset", str],  # type: ignore
        dataset_args: th.Optional[th.Dict[str, th.Any]] = None,
        standardize: bool = False,
    ):
        super().__init__(dataset=dataset, dataset_args=dataset_args, name="Score", standardize=standardize)
        self.data = self.get_data(conversion="tensor")
        print(self.data)
        self.dag, self.order = SCORE(self.data, 0.001, 0.001, 0.001)
        print(self.dag, self.order)
        self.dag = nx.DiGraph(self.dag)

    def estimate_order(self):
        return self.order

    def estimate_dag(self):
        return self.dag
