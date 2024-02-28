# Codes are adopted from the original implementation
# https://github.com/paulrolland1307/SCORE/tree/5e18c73a467428d51486d2f683349dde2607bfe1
# under the GNU Affero General Public License v3.0
# Copy right belongs to the original author https://github.com/paulrolland1307

from source.base import AbstractBaseline  # also adds ocd to sys.path
import networkx as nx
from source.methods.score.stein import SCORE


class Score(AbstractBaseline):
    def __init__(self, standard: bool = False):
        super().__init__(name="SCORE", standard=standard)
        self.dag = None
        self.order = None

    def estimate_order(self):
        if self.order is None:
            data = self.get_samples(conversion="tensor")
            self.dag, self.order = SCORE(data, 0.001, 0.001, 0.001)
            self.dag = nx.DiGraph(self.dag)

    def estimate_dag(self):
        if self.dag is None:
            self.estimate_order()
        return self.dag
