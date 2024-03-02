# Codes are adopted from the original implementation
# https://github.com/vzantedeschi/DAGuerreotype/blob/main/daguerreo/run_model.py
# under the BSD 3-Clause License
# Copyright (c) 2023, Valentina Zantedeschi
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# you should have the main package installed
from daguerreo.models import Daguerro  # type: ignore
from daguerreo import utils  # type: ignore
from daguerreo.args import parse_pipeline_args  # type: ignore
import networkx as nx
import torch
import wandb

from source.base import AbstractBaseline


class DAGuerreo(AbstractBaseline):
    """DAG Learning on the Permutohedron baseline. https://arxiv.org/pdf/2301.11898.pdf"""

    def __init__(
        self,
        num_epochs: int = 5000,
        linear: bool = False,
        seed: int = 42,
        sp_map: bool = False,
        joint: bool = False,
    ):
        super().__init__(name="DAGuerreo")
        self.linear = linear

        # parse args
        arg_parser = parse_pipeline_args()
        self.args = arg_parser.parse_args([])
        self.args.nogpu = False
        self.seed = seed
        self.args.equations = "linear" if self.linear else "nonlinear"
        self.args.num_epochs = num_epochs
        self.args.joint = joint
        self.args.structure = "sp_map" if sp_map else "tk_sp_max"

    @staticmethod
    def _estimate_order_dat(samples, args, seed):
        print(args)
        utils.init_seeds(seed=seed)
        torch.set_default_dtype(torch.double)
        daguerro = Daguerro.initialize(samples, args, args.joint)
        daguerro, samples = utils.maybe_gpu(args, daguerro, samples)
        log_dict = daguerro(samples, utils.AVAILABLE[args.loss], args)
        daguerro.eval()
        _, dags = daguerro(samples, utils.AVAILABLE[args.loss], args)

        estimated_adj = dags[0].detach().cpu().numpy()
        g = nx.DiGraph(estimated_adj)
        # log_dict |= count_accuracy(samples.cpu().detach().numpy(), estimated_adj)
        # print(log_dict)
        orders = list(nx.topological_sort(g))

        return g, orders

    def estimate_order(self):
        samples = self.get_samples(conversion="tensor").double()
        self.args.sparsifier = "none"
        _, orders = self._estimate_order_dat(samples, self.args, self.seed)
        return orders

    def estimate_dag(self):
        samples = self.get_samples(conversion="tensor").double()
        self.args.sparsifier = "l0_ber_ste"
        graph, _ = self._estimate_order_dat(samples, self.args, self.seed)
        return graph
