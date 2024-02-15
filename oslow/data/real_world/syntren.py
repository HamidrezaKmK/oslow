# Codes are adopted from the original implementation
# https://github.com/vzantedeschi/DAGuerreotype/blob/main/daguerreo/data/datasets.py
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

import networkx as nx

from oslow.data import OCDDataset
import pandas as pd
import numpy as np
import os
import typing as th

DEFAULT_SEED = 1234

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def reordering(adj_mat, data):
    rng = np.random.default_rng(DEFAULT_SEED)
    P = rng.permutation(np.eye(adj_mat.shape[0]))
    r = np.nonzero(P)[1]

    adj_mat = P @ adj_mat @ P.T
    data = data[:, r]
    return adj_mat, data


class SyntrenOCDDataset(OCDDataset):
    def __init__(
        self,
        data_id: int,
        standardization: bool = False,
        reject_outliers_n_far_from_mean: th.Optional[float] = None,
    ):
        """Args:
        data_id: The id of the dataset to load (from 0 to 9)
        """
        # load csv file into pandas dataframe
        data = np.load(os.path.join(_DATA_DIR, "syntren", f"data{data_id+1}.npy"))
        adj_mat = np.load(os.path.join(_DATA_DIR, "syntren", f"DAG{data_id+1}.npy"))

        adj_mat, data = reordering(
            adj_mat, data
        )  # Since the default order is [1, ..., d]
        graph = nx.DiGraph(adj_mat)

        df = pd.DataFrame(data)

        super().__init__(
            samples=df,
            dag=graph,
            name=f"Syntren-{data_id}",
            standardization=standardization,
            reject_outliers_n_far_from_mean=reject_outliers_n_far_from_mean,
        )
