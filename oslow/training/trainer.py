import torch

from oslow.config import (
    GumbelSinkhornStraightThroughConfig,
    GumbelTopKConfig,
    SoftSinkhornConfig,
    ContrastiveDivergenceConfig,
    SoftSortConfig,
    BirkhoffConfig,
)

import numpy as np
import wandb
import matplotlib.pyplot as plt
import networkx as nx

from torch.utils.data import DataLoader
from typing import Union, Callable, Iterable, Optional, Literal
from oslow.models.oslow import OSlow
from oslow.training.permutation import (
    GumbelTopK,
    ContrastiveDivergence,
    GumbelSinkhornStraightThrough,
    SoftSort,
    SoftSinkhorn,
)
from oslow.visualization.birkhoff import visualize_birkhoff_polytope
from oslow.evaluation import backward_relative_penalty
import os
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)



class Trainer:
    def __init__(
        self,
        model: OSlow,
        dag: nx.DiGraph,
        flow_dataloader: DataLoader,
        perm_dataloader: DataLoader,
        flow_optimizer: Callable[[Iterable], torch.optim.Optimizer],
        permutation_optimizer: Callable[[Iterable], torch.optim.Optimizer],
        flow_frequency: int,
        permutation_frequency: int,
        max_epochs: int,
        flow_lr_scheduler: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ],
        permutation_lr_scheduler: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ],
        permutation_learning_config: Union[
            GumbelSinkhornStraightThroughConfig,
            GumbelTopKConfig,
            SoftSinkhornConfig,
            ContrastiveDivergenceConfig,
        ],
        birkhoff_config: Optional[BirkhoffConfig] = None,
        temperature: float = 1.0,
        temperature_scheduler: Literal['constant',
                                       'linear', 'exponential'] = 'constant',
        device: str = "cpu",

        perform_final_buffer_search: bool = False,
    ):
        self.device = device
        self.max_epochs = max_epochs
        self.model = model.to(device)
        permutation_learning_kwargs = permutation_learning_config.model_dump()
        method = permutation_learning_kwargs.pop("method")
        if method == "contrastive-divergence":
            self.permutation_learning_module = ContrastiveDivergence(
                model.in_features, **permutation_learning_kwargs
            ).to(device)
        elif method == "gumbel-top-k":
            self.permutation_learning_module = GumbelTopK(
                model.in_features, **permutation_learning_kwargs
            ).to(device)
        elif method == "straight-through-sinkhorn":
            self.permutation_learning_module = GumbelSinkhornStraightThrough(
                model.in_features, **permutation_learning_kwargs
            ).to(device)
        elif method == "soft-sort":
            self.permutation_learning_module = SoftSort(
                model.in_features, **permutation_learning_kwargs
            ).to(device)
        elif method == "soft-sinkhorn":
            self.permutation_learning_module = SoftSinkhorn(
                model.in_features, **permutation_learning_kwargs
            ).to(device)
        else:
            # TODO: update and add other baselines for ablation study
            raise ValueError(
                "permutation_learning_config must be of type GumbelSinkhornStraightThroughConfig or ContrastiveDivergenceConfig"
            )
        self.perform_final_buffer_search = perform_final_buffer_search
        if self.perform_final_buffer_search and not hasattr(self.permutation_learning_module, "update_buffer"):
            raise ValueError(
                "The permutation learning module must have a buffer to perform final buffer search"
            )
        self.flow_dataloader = flow_dataloader
        self.perm_dataloader = perm_dataloader
        self.permutation_learning_config = permutation_learning_config
        self.birkhoff_config = birkhoff_config
        self.flow_optimizer_fn = flow_optimizer
        self.flow_optimizer = flow_optimizer(self.model.parameters())
        self.permutation_optimizer = permutation_optimizer(
            self.permutation_learning_module.parameters()
        )
        self.flow_frequency = flow_frequency
        self.permutation_frequency = permutation_frequency
        self.flow_scheduler = flow_lr_scheduler(self.flow_optimizer)
        self.permutation_scheduler = permutation_lr_scheduler(
            self.permutation_optimizer
        )
        self.dag = dag
        self.initial_temperature = temperature
        self.temperature_scheduler = temperature_scheduler
        self.perm_step_count = 0
        self.flow_step_count = 0

        # TODO create checkpointing
        self.checkpointing = None

    def get_temperature(self, epoch: int):
        if self.temperature_scheduler == "constant":
            return self.initial_temperature
        # start from initial_temperature and decrease it to 0
        if self.temperature_scheduler == "linear":
            return self.initial_temperature * (1 - (0 if epoch == 0 else epoch / (self.max_epochs - 1)))
        if self.temperature_scheduler == "exponential":
            return self.initial_temperature * (0.1 ** (0 if epoch == 0 else epoch / (self.max_epochs - 1)))

    def flow_train_step(self, temperature: float = 1.0, lbl: str = "flow_ensemble/"):
        self.permutation_learning_module._gamma.requires_grad = False
        avg_loss = []
        for batch in self.flow_dataloader:
            batch = batch.to(self.model.device)
            self.flow_optimizer.zero_grad()
            loss = self.permutation_learning_module.flow_learning_loss(
                model=self.model, batch=batch, temperature=temperature
            )
            loss.backward()
            self.flow_optimizer.step()
            self.flow_step_count += 1
            wandb.log({f"{lbl}step": self.flow_step_count})
            wandb.log({f"{lbl}loss": loss.item()})
            avg_loss.append(loss.item())
        self.permutation_learning_module._gamma.requires_grad = True
        return np.array(avg_loss).mean()

    def permutation_train_step(self, temperature: float = 1.0):
        # stop gradient model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        for batch in self.perm_dataloader:
            batch = batch.to(self.model.device)
            self.permutation_optimizer.zero_grad()
            loss = self.permutation_learning_module.permutation_learning_loss(
                model=self.model, batch=batch, temperature=temperature
            )
            loss.backward()
            self.permutation_optimizer.step()
            self.perm_step_count += 1
            wandb.log({"permutation/step": self.perm_step_count})
            wandb.log({"permutation/loss": loss.item()})
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
        return loss

    def log_evaluation(self, temperature: float = 1.0):

        permutation = self.permutation_learning_module.get_best(
            temperature=temperature)
        permutation = permutation.argmax(dim=-1).cpu().numpy().tolist()
        backward_penalty = backward_relative_penalty(permutation, self.dag)
        wandb.log({"permutation/best_backward_penalty": backward_penalty})

        sampled_permutations = self.permutation_learning_module.sample_hard_permutations(
            100, gumbel_std=temperature
        )
        sampled_permutations_unique, counts = torch.unique(
            sampled_permutations, dim=0, return_counts=True)
        sm = 0
        for perm, c in zip(sampled_permutations_unique, counts):
            backward_penalty = backward_relative_penalty(
                perm.argmax(dim=-1).cpu().numpy().tolist(), self.dag)
            sm += c * backward_penalty
        wandb.log(
            {"permutation/avg_backward_penalty": sm/100}
        )

    def train(self):
        self.model.train()
        self.permutation_learning_module.train()

        for epoch in range(self.max_epochs):
            # reinsitialize the parameters of self.model
            self.model = self.model.to(self.device)

            wandb.log({"permutation/temperature": self.get_temperature(epoch)})
            wandb.log({"epoch": epoch})

            for i in range(self.flow_frequency):
                loss = self.flow_train_step(
                    temperature=self.get_temperature(epoch))
                # logging.info(
                #     f"Flow step {epoch * self.flow_frequency + i} / {self.max_epochs * self.flow_frequency}, flow loss: {loss.item()}"
                # )
                if isinstance(
                    self.flow_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.flow_scheduler.step(loss.item())
                else:
                    self.flow_scheduler.step()

            if hasattr(self.permutation_learning_module, "update_buffer"):
                self.permutation_learning_module.update_buffer(
                    dloader=self.perm_dataloader,
                    model=self.model,
                    temperature=self.get_temperature(epoch),
                )
            for j in range(self.permutation_frequency):
                loss = self.permutation_train_step(
                    temperature=self.get_temperature(epoch))

                # logging.info(
                #     f"Permutation step {epoch * self.permutation_frequency + i} / {self.max_epochs * self.permutation_frequency}, permutation loss: {loss.item()}"
                # )
                if isinstance(
                    self.permutation_scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.permutation_scheduler.step(loss.item())
                else:
                    self.permutation_scheduler.step()

                # log the evaluation metrics
                self.log_evaluation(temperature=self.get_temperature(epoch))

                # log the Birkhoff polytope
                if (
                    self.model.in_features <= 4
                    and self.birkhoff_config
                    and (
                        epoch == 0 or
                        epoch == self.max_epochs - 1 or
                        (j + epoch * self.permutation_frequency + 1)
                        % self.birkhoff_config.frequency
                        == 0
                    )
                ):
                    batch = next(iter(self.perm_dataloader))
                    img = visualize_birkhoff_polytope(
                        permutation_model=self.permutation_learning_module,
                        num_samples=self.birkhoff_config.num_samples,
                        data=batch,
                        flow_model=self.model,
                        device=self.device,
                        print_legend=self.birkhoff_config.print_legend,
                        dag=self.dag,
                        temperature=self.get_temperature(epoch),
                    )
                    wandb.log(
                        {
                            "permutation/birkhoff": wandb.Image(
                                img, caption="Birkhoff Polytope"
                            )
                        }
                    )

        final_phase_buffer_size = len(
            self.permutation_learning_module.permutation_buffer)
        final_phase_epoch_count = self.max_epochs * \
            (self.permutation_frequency +
             self.flow_frequency) // final_phase_buffer_size
        if final_phase_epoch_count > 0 and final_phase_buffer_size > 0 and hasattr(self.permutation_learning_module, "update_buffer"):
            cap = min(10, len(
                self.permutation_learning_module.permutation_buffer))
            candidate_permutations = self.permutation_learning_module.permutation_buffer[:cap].cpu(
            ).numpy()
            best_avg_loss = None
            best_perm = None

            def tmp_fn(perm_list, best_avg_loss, best_perm, lbl):
                self.permutation_learning_module.fix_permutation(perm_list)
                self.model.reinitialize()
                self.flow_optimizer = self.flow_optimizer_fn(
                    self.model.parameters())
                for _ in range(final_phase_epoch_count):
                    avg_loss = self.flow_train_step(
                        temperature=0.0, lbl=f"{lbl}/{'-'.join(map(str, perm_list))}-")
                if best_avg_loss is None or avg_loss < best_avg_loss:
                    best_avg_loss = avg_loss
                    best_perm = perm_list
                return best_avg_loss, best_perm

            for perm in candidate_permutations:
                perm_list = perm.argmax(-1).tolist()
                best_avg_loss, best_perm = tmp_fn(
                    perm_list, best_avg_loss, best_perm, lbl="buffered_flows")

            _, _ = tmp_fn(best_perm, best_avg_loss,
                          best_perm, lbl="final_flow")

            self.log_evaluation(temperature=0.0)
