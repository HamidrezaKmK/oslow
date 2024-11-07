import torch

import numpy as np
import wandb
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

from oslow.training.utils import turn_into_matrix

from torch.utils.data import DataLoader
from typing import Union, Callable, Iterable, Optional, Literal, List
from oslow.models.oslow import OSlow
from oslow.visualization.birkhoff import visualize_birkhoff_polytope
from oslow.evaluation import backward_relative_penalty
import os


class EnsembleTrainer:
    """
    This class takes in a permutation learning module which is fixed and generates a set of permutations.
    The flow model is then used on this distribution of permutations to learn an ensemble flow model.

    Finally, during training, a scatterplot is generated juxtaposing
    """

    def __init__(
        self,
        model: OSlow,
        dag: nx.DiGraph,
        flow_dataloader: DataLoader,
        flow_ensemble_dataloader: DataLoader,
        flow_optimizer: Callable[[Iterable], torch.optim.Optimizer],
        max_epochs: int,
        flow_lr_scheduler: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ],
        perm_list: List[List[int]],
        logging_scatterplot_frequency: int = 1,
        device: str = "cpu",
    ):
        """
        Args:
            dag: The causal DAG

            model: The likelihood flow model that is used to determine whether an ordering is better than another
            flow_dataloader: The dataloader used for flow learning
            flow_optimizer: The optimizer for the flow learning, this would influence the weights of the flow model
            max_epochs: The maximum number of epochs
            flow_lr_scheduler: The learning rate scheduler for the flow learning
            perm_list: A list of permutations which is sampled from.
            logging_scatterplot_frequency: The frequency at which the scatterplot is logged onto wandb

            device: The device to be used

        """
        self.device = device
        self.max_epochs = max_epochs
        self.model = model.to(device)

        self.perm_list = []
        self.cbc_penalties = []
        for perm in perm_list:
            self.cbc_penalties.append(
                backward_relative_penalty(perm=perm, dag=dag)
            )
            self.perm_list.append(turn_into_matrix(torch.IntTensor(perm)))
        self.perm_list = torch.stack(self.perm_list).to(device)
        self.nlls = []

        self.flow_ensemble_dataloader = flow_ensemble_dataloader
        self.flow_dataloader = flow_dataloader
        self.flow_optimizer_instantiate = flow_optimizer
        self.flow_lr_scheduler_instantiate = flow_lr_scheduler
        self.logging_scatterplot_frequency = logging_scatterplot_frequency

        # save the dag for evaluation
        self.dag = dag
        self.perm_step_count = 0
        self.flow_step_count = 0


    def train(self, epoch: int, permutations: torch.Tensor, lbl: str):
        avg_loss = []
        for batch in self.flow_ensemble_dataloader:
            batch = batch.to(self.model.device)
            b_size = batch.shape[0]

            # uniformly sample from permutation based on b_size
            perm_mats = permutations[torch.randint(0, len(permutations), (b_size,))]

            # perm_mats = permutations.repeat_interleave(b_size, dim=0)
            # batch_repeated = batch.repeat(len(permutations), 1)

            self.flow_optimizer.zero_grad()

            log_probs = self.model.log_prob(batch, perm_mat=perm_mats)
            loss = -torch.mean(log_probs)

            loss.backward()
            self.flow_optimizer.step()
            self.flow_step_count += 1
            wandb.log({f"{lbl}/step": self.flow_step_count})
            wandb.log({f"{lbl}/loss": loss.item()})
            avg_loss.append(loss.item())

        # Perform a learning rate scheduling step
        if isinstance(self.flow_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.flow_scheduler.step(sum(avg_loss) / len(avg_loss))
        else:
            self.flow_scheduler.step()

    def visualize_scatterplot(self):

        img_data = None

        try:
            # plot the log_prob values of the closest references using the data if available
            fig, ax = plt.subplots()
            ax.scatter(
                self.cbc_penalties,
                self.nlls,
                label=f"nll vs. cbc",
            )
            ax.set_ylabel("negative log-likelihood (NLL)")
            ax.set_xlabel("Causal Backward Count (CBC) penalty")
            # draw everything to the figure for conversion
            fig.canvas.draw()
            # convert the figure to a numpy array
            img_data = np.fromstring(
                fig.canvas.tostring_rgb(), dtype=np.uint8, sep=""
            )
            img_data = img_data.reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
        finally:
            plt.close()

        wandb.log(
            {
                f"scatterplot/nll_vs_cbc": wandb.Image(
                    img_data
                )
            }
        )

    def get_log_prob(self, perm: torch.Tensor) -> float:
        log_probs = []
        for batch in self.flow_dataloader:
            batch = batch.to(self.model.device)
            b_size = batch.shape[0]
            perm_repeated = perm.unsqueeze(0).repeat_interleave(b_size, dim=0)
            log_probs.append(
                self.model.log_prob(batch, perm_mat=perm_repeated).mean().item()
            )
        return sum(log_probs) / len(log_probs)

    def run(self):
        self.model.train()
        self.model = self.model.to(self.device)

        self.flow_optimizer = self.flow_optimizer_instantiate(self.model.parameters())
        self.flow_scheduler = self.flow_lr_scheduler_instantiate(self.flow_optimizer)

        for epoch in tqdm(range(self.max_epochs)):
            self.train(epoch, self.perm_list, lbl="ensemble")

            if (epoch + 1) % self.logging_scatterplot_frequency == 0:
                self.nlls = []
                for perm in self.perm_list:
                    self.nlls.append(-self.get_log_prob(perm))
                self.visualize_scatterplot()

            wandb.log({"epoch/ensemble": epoch})

