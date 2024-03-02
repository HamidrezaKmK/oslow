# Codes are adopted from the original implementation
# https://github.com/sharpenb/Differentiable-DAG-Sampling
# you should have the main package installed

import torch

import networkx as nx
from source.base import AbstractBaseline

from .probabilistic_dag import ProbabilisticDAG as FixedProbabilisticDAG

import os, sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_DIR)
import src.probabilistic_dag_model.probabilistic_dag  # type: ignore

setattr(
    src.probabilistic_dag_model.probabilistic_dag,
    "ProbabilisticDAG",
    FixedProbabilisticDAG,
)
from src.probabilistic_dag_model.probabilistic_dag_autoencoder import (  # type: ignore
    ProbabilisticDAGAutoencoder,
)
from src.probabilistic_dag_model.train_probabilistic_dag_autoencoder import (  # type: ignore
    train_autoencoder,
)


class VI_DP_DAG(AbstractBaseline):
    """DIFFERENTIABLE DAG SAMPLING from https://arxiv.org/pdf/2203.08509.pdf"""

    def __init__(
        self,
        seed: int = 42,
        method="topk",  # topk or sinkhorn
        max_epochs=100,
    ):
        super().__init__(name="VI-DP-DAG")

        # parse args
        self.model_args = {
            "output_dim": 1,
            "ma_hidden_dims": [16, 16, 16],
            "ma_architecture": "linear",
            "ma_fast": False,
            "pd_initial_adj": None,
            "pd_temperature": 1.0,
            "pd_hard": True,
            "pd_order_type": method,
            "pd_noise_factor": 1.0,
            "ma_lr": 1e-3,
            "pd_lr": 1e-2,
            "loss": "ELBO",
            "regr": 0.01,
            "prior_p": 0.01,
            "seed": seed,
        }
        # number of dags to sample to use as the prediction dag
        self.num_sample_dags = 1000

        self.max_epochs = max_epochs
        self.patience = 10
        self.frequency = 2
        self.model_path = os.path.join(_DIR, "VI_DP_DAG_saved_model")
        self.model = None
        self.val_size = 0.1
        self.batch_size = 64

    def train_and_predict(self):
        if self.model is None:
            samples = self.get_samples(conversion="tensor").float()
            self.model_args["input_dim"] = samples.shape[1]
            val_size = int(self.val_size * len(samples))
            train_data, val_data = torch.utils.data.random_split(
                samples,
                [
                    len(samples) - val_size,
                    val_size,
                ],
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_data, batch_size=self.batch_size, shuffle=True
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_data, batch_size=self.batch_size, shuffle=True
            )
            self.model = ProbabilisticDAGAutoencoder(**self.model_args)
            train_losses, val_losses, train_mse, val_mse = train_autoencoder(
                model=self.model,
                train_loader=train_dataloader,
                val_loader=val_dataloader,
                max_epochs=self.max_epochs,
                frequency=self.frequency,
                patience=self.patience,
                model_path=self.model_path,
            )

        # predict
        self.model.eval()
        if self.num_sample_dags == 0:  # use the learned mask itself
            if self.model.pd_initial_adj is None:  # DAG is learned
                prob_mask = self.model.probabilistic_dag.get_prob_mask()
            else:  # DAG is fixed
                prob_mask = self.model.pd_initial_adj
            return nx.DiGraph(prob_mask.detach().cpu().numpy())

        # sample dags
        dags = torch.stack(
            [
                self.model.probabilistic_dag.sample()
                for i in range(self.num_sample_dags)
            ],
            dim=0,
        )
        # count how many times each dag is sampled
        uniques, counts = torch.unique(dags, return_counts=True, dim=0)
        # pick the most sampled dag
        return nx.DiGraph(uniques[counts.argmax()].detach().cpu().numpy())

    def estimate_order(self):
        dag = self.train_and_predict()
        return list(nx.topological_sort(dag))

    def estimate_dag(self):
        dag = self.train_and_predict()
        return dag
