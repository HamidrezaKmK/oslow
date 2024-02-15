# Codes are adopted from the original implementation
# https://github.com/sharpenb/Differentiable-DAG-Sampling
# you should have the main package installed

import networkx as nx
import typing as th
from source.base import AbstractBaseline

from .probabilistic_dag import ProbabilisticDAG as FixedProbabilisticDAG
import src.probabilistic_dag_model.probabilistic_dag

setattr(src.probabilistic_dag_model.probabilistic_dag, "ProbabilisticDAG", FixedProbabilisticDAG)
from src.probabilistic_dag_model.probabilistic_dag_autoencoder import ProbabilisticDAGAutoencoder
from src.probabilistic_dag_model.train_probabilistic_dag_autoencoder import train_autoencoder
from lightning_toolbox.data import DataModule
import torch


class SamplesDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DifferentiableDagSampling(AbstractBaseline):
    """DIFFERENTIABLE DAG SAMPLING from https://arxiv.org/pdf/2203.08509.pdf"""

    def __init__(
        self,
        dataset: th.Union["OCDDataset", str],  # type: ignore
        dataset_args: th.Optional[th.Dict[str, th.Any]] = None,
        # hyperparameters
        seed: int = 42,
        ma_hidden_dims=[16, 16, 16],  # Same
        ma_architecture="linear",  # Same
        ma_fast=False,  # Same
        pd_initial_adj=None,  # Same
        pd_temperature=1.0,  # Same
        pd_hard=True,  # same
        pd_order_type="topk",  # same
        pd_noise_factor=1.0,  # same
        max_epochs=100,  # input
        patience=10,  # same
        frequency=2,  # same
        ma_lr=1e-3,  # same
        pd_lr=1e-2,  # same
        loss="ELBO",  # same
        regr=0.01,  # same
        prior_p=0.01,  # same
        model_path="saved_model",
        # prediction args OURS
        num_sample_dags: int = 1000,  # 0 for using the learned mask itself
        # data args OURS
        standardize: bool = False,
        val_size: float = 0.1,
        batch_size: int = 64,
        **datamodule_args,
    ):
        super().__init__(
            dataset=dataset, dataset_args=dataset_args, name="DifferentiableDagSampling", standardize=standardize
        )

        # parse args
        self.samples = self.get_data(conversion="tensor").float()
        self.model_args = {
            "input_dim": self.samples.shape[1],
            "output_dim": 1,
            "ma_hidden_dims": ma_hidden_dims,
            "ma_architecture": ma_architecture,
            "ma_fast": ma_fast,
            "pd_initial_adj": pd_initial_adj,
            "pd_temperature": pd_temperature,
            "pd_hard": pd_hard,
            "pd_order_type": pd_order_type,
            "pd_noise_factor": pd_noise_factor,
            "ma_lr": ma_lr,
            "pd_lr": pd_lr,
            "loss": loss,
            "regr": regr,
            "prior_p": prior_p,
            "seed": seed,
        }
        self.num_sample_dags = num_sample_dags  # number of dags to sample to use as the prediction dag

        self.max_epochs = max_epochs
        self.patience = patience
        self.frequency = frequency
        self.model_path = model_path
        self.model = None
        self.datamodule_args = {**datamodule_args, "val_size": val_size, "batch_size": batch_size}

    def train_and_predict(self):
        if self.model is not None:
            return self.predict()
        dm = DataModule(dataset=SamplesDataset(self.samples), **self.datamodule_args)
        dm.setup("fit")
        self.model = ProbabilisticDAGAutoencoder(**self.model_args)
        train_losses, val_losses, train_mse, val_mse = train_autoencoder(
            model=self.model,
            train_loader=dm.train_dataloader(),
            val_loader=dm.val_dataloader(),
            max_epochs=self.max_epochs,
            frequency=self.frequency,
            patience=self.patience,
            model_path=self.model_path,
        )
        return self.predict()

    def predict(self):
        if self.model is None:
            return self.train_and_predict()  # need to train first
        self.model.eval()
        if self.num_sample_dags == 0:  # use the learned mask itself
            if self.model.pd_initial_adj is None:  # DAG is learned
                prob_mask = self.model.probabilistic_dag.get_prob_mask()
            else:  # DAG is fixed
                prob_mask = self.model.pd_initial_adj
            return nx.DiGraph(prob_mask.detach().cpu().numpy())

        # sample dags
        dags = torch.stack([self.model.probabilistic_dag.sample() for i in range(self.num_sample_dags)], dim=0)
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
