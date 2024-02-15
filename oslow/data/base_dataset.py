import torch
import pandas as pd
from typing import Union, Optional
import numpy as np
import networkx as nx
from sklearn.neighbors import KernelDensity


class OCDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        dag: nx.DiGraph,
        name: Optional[str] = None,
        standard: bool = False,
    ):
        """
        Args:
            samples: a pandas DataFrame
            dag: a networkx DiGraph
            name: name of the dataset (optional, used for printing)
            standard: whether to standardize the samples or not, either ways self.samples_statistic
                            contains the mean and std of each column as a dictionary
        """
        self.name = name
        # set the dag
        self.dag = dag

        if samples is None:
            return

        # standardize all the columns
        self.samples_statistics = {}
        # sort columns of samples alphabetically
        self.samples = samples.reindex(sorted(samples.columns), axis=1)

        if standard:
            for col in self.samples.columns:
                avg = self.samples[col].mean()
                std = self.samples[col].std()
                self.samples[col] = (self.samples[col] - avg) / (std + 1e-8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # return the idx-th sample, as a jnp.array
        ret = self.samples.iloc[idx].values
        # turn ret into float32
        ret = ret.astype(np.float32)
        return ret
