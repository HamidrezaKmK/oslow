# Codes are adopted from the original implementation
# https://github.com/paulrolland1307/SCORE/tree/5e18c73a467428d51486d2f683349dde2607bfe1
# under the GNU Affero General Public License v3.0
# Copy right belongs to the original author https://github.com/paulrolland1307

from source.base import AbstractBaseline  # also adds ocd to sys.path
from source.utils import full_DAG
import typing as th
import numpy as np
import networkx as nx
from source.methods.lsnm.loci import loci


class LSNM(AbstractBaseline):
    def __init__(
        self,
        dataset: th.Union["OCDDataset", str],  # type: ignore
        dataset_args: th.Optional[th.Dict[str, th.Any]] = None,
        # hyperparameters
        standardize: bool = False,
        verbose: bool = True,
        independence_test: bool = True,
        neural_network: bool = True,
        n_steps: int = 1000,
        independence_eps: float = 0.01,
    ):
        super().__init__(dataset=dataset, dataset_args=dataset_args, name="LSNM", standardize=standardize)
        self.verbose = verbose
        self.independence_test = independence_test
        self.neural_network = neural_network
        self.n_steps = n_steps
        self.independence_eps = independence_eps
        self.data = self.get_data(conversion="numpy")

    def estimate_order(self):
        data = self.data
        dag = np.zeros((data.shape[1], data.shape[1]))
        for i in range(data.shape[1]):
            print(f"Estimating order for variable {i}")
            for j in range(i, data.shape[1]):
                score = loci(
                    x=data[:, i],
                    y=data[:, j],
                    independence_test=self.independence_test,
                    neural_network=self.neural_network,
                    return_function=False,
                    n_steps=self.n_steps,
                )
                if self.independence_eps and abs(score) < self.independence_eps:
                    continue
                dag[i, j] = 1 if score > 0 else 0
                dag[j, i] = 1 if score < 0 else 0
        self.dag = dag
        # compute the topological order of the dag
        g = nx.DiGraph(dag)
        order = list(nx.topological_sort(g))
        return order

    def estimate_dag(self):
        return nx.DiGraph(self.dag)
