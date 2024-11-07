import torch

from .permutation_learning.baseline_methods import PermutationLearningModule
import numpy as np
import wandb
import matplotlib.pyplot as plt
import networkx as nx

from torch.utils.data import DataLoader
from typing import Union, Callable, Iterable, Optional, Literal
from oslow.models.oslow import OSlow
from oslow.visualization.birkhoff import visualize_birkhoff_polytope
from oslow.evaluation import backward_relative_penalty
import os
import logging


class Trainer:
    """
    This class is an alternating trainer for causal order learning.
    It takes in the likelihood model OSlow which is an ensemble of flows and acts on a specific permutation and
    a batch of data to produce the best possible likelihood given that ordering for the input batch of data given.
    
    The trainer also contains a permutation learning module which parameterizes a distribution over the
    space of permutations. The permutation learning module is trained to sample permutations that are
    have the best likelihood on the input data.
    
    Rough sketch of the training process:
    
    1. Train the flow model with the permutation learning module frozen: the learn_flow method
        This is done for flow_frequency number of times
    2. Train the permutation learning module with the flow model frozen: the learn_permutation method
        This is done for permutation_frequency number of times
    3. Log the evaluation metrics: the log_evaluation method
        This includes the average number of backward edges for the best permutation
    4. Log the Birkhoff polytope if applicable: the log_polytope method
    5. Repeat 1-4 for the number of epochs
    6. Perform a final phase if applicable: the final_phase method
        This includes training fixed permutation models once a set of best permutations are found
    """
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
        permutation_learning_module: Callable[[int], PermutationLearningModule],
        temperature: float = 1.0,
        temperature_scheduler: Literal['constant',
                                       'linear', 'exponential'] = 'constant',
        device: str = "cpu",
        
        birkhoff_plot_frequency: Optional[int] = None,
        birkhoff_plot_num_samples: Optional[int] = None,
        birkhoff_plot_print_legend: bool = False,
        
        checkpointing: Optional[Union[str, os.PathLike]] = None,

    ):
        """
        Args:
            dag: The causal DAG
            
            model: The likelihood flow model that is used to determine whether an ordering is better than another
            flow_dataloader: The dataloader used for flow learning
            perm_dataloader: The dataloader used for permutation learning (batch size might differ from that of flow learning)
            flow_optimizer: The optimizer for the flow learning, this would influence the weights of the flow model 
            permutation_optimizer: The optimizer for the permutation learning, this would influence the weights of the permutation learning module
            flow_frequency: In each epoch, how many times the flow learning is performed
            permutation_frequency: In each epoch, how many times the permutation learning is performed
            max_epochs: The maximum number of epochs
            flow_lr_scheduler: The learning rate scheduler for the flow learning
            permutation_lr_scheduler: The learning rate scheduler for the permutation learning
            
            
            permutation_learning_module: 
                The permutation learning module which has trainable parameters directly influencing the order-learning aspect
                These modules all inherit the ones in oslow.permutation.PermutationLearningModule
            temperature: 
                The permutation learning module samples permutations on a Boltzmann distribution which is controlled by the temperature
                this is the base temperature which is used.
            temperature_scheduler:
                One can optionally anneal the temperature over time. This can be done in three ways:
                - constant: The temperature remains constant
                - linear: The temperature linearly decreases from the initial temperature to 0
                - exponential: The temperature decreases exponentially from the initial temperature to 0
                
            
            birkhoff_plot_frequency: 
                The frequency of which the Birkhoff polytope is being plotted, defaults to None which means there's no need for visualization at all
            birkhoff_plot_num_samples:
                The number of samples used for visualizing the Birkhoff polytope, defaults to None which means there's no need for visualization at all
            birkhoff_plot_print_legend:
                The boolean value to print the legend on the Birkhoff polytope, defaults to False
            
            device: The device to be used
            
            checkpointing: Not implemented yet!
        """
        self.device = device
        self.max_epochs = max_epochs
        self.model = model.to(device)
        
        # instantiate the permutation learning module for a number of given features
        self.permutation_learning_module = permutation_learning_module(
            model.in_features
        ).to(device)
        
        self.flow_dataloader = flow_dataloader
        self.perm_dataloader = perm_dataloader
        
        # configurations of logging the Birkhoff polytope
        self.has_birkhoff = birkhoff_plot_frequency is not None
        self.birkhoff_plot_frequency = birkhoff_plot_frequency
        self.birkhoff_plot_num_samples = birkhoff_plot_num_samples
        self.birkhoff_plot_legend = birkhoff_plot_print_legend
        
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
        # save the dag for evaluation
        self.dag = dag
        self.initial_temperature = temperature
        self.temperature_scheduler = temperature_scheduler
        self.perm_step_count = 0
        self.flow_step_count = 0

        # TODO: add checkpointing
        self.checkpointing = None
    
    def get_temperature(self, epoch: int):
        """
        Returns the temperature based on the temperature scheduler
        
        It is assumed that the temperature is annealed over time with `epoch` changing from 0
        to `max_epochs - 1`
        """
        if self.temperature_scheduler == "constant":
            return self.initial_temperature
        # start from initial_temperature and decrease it to 0
        if self.temperature_scheduler == "linear":
            return self.initial_temperature * (1 - (0 if epoch == 0 else epoch / (self.max_epochs - 1)))
        if self.temperature_scheduler == "exponential":
            return self.initial_temperature * (0.1 ** (0 if epoch == 0 else epoch / (self.max_epochs - 1)))

    def log_evaluation(self, temperature: float = 1.0):
        """
        This function logs evaluation metrics based on the permutation model.
        
        For example, it samples a set of permutations and then looks at the full causal
        DAG to see that for each permutation, how many backward edges are there.
        Finally, it logs onto wandb the average number of backward edges.
        """

        permutation = self.permutation_learning_module.get_best(
            temperature=temperature)
        permutation = permutation.argmax(dim=0).cpu().numpy().tolist()
        backward_penalty = backward_relative_penalty(permutation, self.dag)
        wandb.log({"evaluation/best_backward_penalty": backward_penalty})

        sampled_permutations = self.permutation_learning_module.sample_permutations(
            100, gumbel_std=temperature
        )
        sampled_permutations_unique, counts = torch.unique(
            sampled_permutations, dim=0, return_counts=True)
        sm = 0
        for perm, c in zip(sampled_permutations_unique, counts):
            backward_penalty = backward_relative_penalty(
                perm.argmax(dim=0).cpu().numpy().tolist(), self.dag)
            sm += c * backward_penalty
        wandb.log(
            {"evaluation/avg_backward_penalty": sm/100}
        )
    
    def log_polytope(self, epoch: int):
        # log the birkhoff polytope on to wandb
        batch = next(iter(self.perm_dataloader))
        img = visualize_birkhoff_polytope(
            permutation_model=self.permutation_learning_module,
            num_samples=self.birkhoff_plot_num_samples,
            data=batch,
            flow_model=self.model,
            device=self.device,
            print_legend=self.birkhoff_plot_legend,
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

    def learn_flow(self, epoch: int):
        for _ in range(self.flow_frequency):
            # freeze the parameters of the permutation learning module
            self.permutation_learning_module.freeze()
            avg_loss = []
            for batch in self.flow_dataloader:
                batch = batch.to(self.model.device)
                self.flow_optimizer.zero_grad()
                # perform a flow learning step by sampling permutations from the permutation learning module
                # and using those to feed into the model
                loss = self.permutation_learning_module.flow_learning_loss(
                    model=self.model, batch=batch, temperature=self.get_temperature(epoch),
                )
                loss.backward()
                self.flow_optimizer.step()
                self.flow_step_count += 1
                wandb.log({f"flow_ensemble/step": self.flow_step_count})
                wandb.log({f"flow_ensemble/loss": loss.item()})
                avg_loss.append(loss.item())
            # unfreeze the parameters of the permutation learning module
            self.permutation_learning_module.unfreeze()
            
            # Perform a learning rate scheduling step
            if isinstance(
                self.flow_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.flow_scheduler.step(sum(avg_loss) / len(avg_loss))
            else:
                self.flow_scheduler.step()

    def learn_permutation(self, epoch: int):
        for _ in range(self.permutation_frequency):
            # stop gradient model
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
            avg_loss = []
            for batch in self.perm_dataloader:
                batch = batch.to(self.model.device)
                self.permutation_optimizer.zero_grad()
                loss = self.permutation_learning_module.permutation_learning_loss(
                    model=self.model, batch=batch, temperature=self.get_temperature(epoch)
                )
                loss.backward()
                self.permutation_optimizer.step()
                self.perm_step_count += 1
                wandb.log({"permutation/step": self.perm_step_count})
                wandb.log({"permutation/loss": loss.item()})
                avg_loss.append(loss.item())
                
            # return gradients for the model
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()
            
            if isinstance(
                self.permutation_scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                self.permutation_scheduler.step(sum(avg_loss) / len(avg_loss))
            else:
                self.permutation_scheduler.step()

    def final_phase(self):
        # a final phase if applicable
        # in the case of a standard trainer, there is no final phase
        print("No final phase!")
    
    def train(self):
        self.model.train()
        self.permutation_learning_module.train()

        for epoch in range(self.max_epochs):
            # reinsitialize the parameters of self.model
            self.model = self.model.to(self.device)

            wandb.log({"permutation/temperature": self.get_temperature(epoch)})
            wandb.log({"epoch": epoch})
            
            # based on the permutation model, train a flow ensemble with Oslow
            self.learn_flow(epoch)
            
            # log the evaluation metrics
            self.log_evaluation(temperature=self.get_temperature(epoch))
            
            # based on the likelihood model, optimize the permutations
            self.learn_permutation(epoch)
            
            # log the Birkhoff polytope
            if (
                self.model.in_features <= 4
                and self.has_birkhoff
                and (
                    epoch == 0 or
                    epoch == self.max_epochs - 1 or
                    (epoch + 1) % self.birkhoff_plot_frequency == 0
                )
            ):
                self.log_polytope(epoch)

            if epoch == self.max_epochs - 1:
                self.log_evaluation(temperature=0.0)

        self.final_phase()

    