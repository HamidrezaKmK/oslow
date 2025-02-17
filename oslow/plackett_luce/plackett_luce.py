import torch

import wandb
import networkx as nx
from tqdm import tqdm

from torch.utils.data import DataLoader
from typing import Callable, Iterable
from oslow.models.oslow import OSlow


from oslow.training.utils import listperm2matperm
from oslow.evaluation import backward_relative_penalty


@torch.no_grad()
def sample_plackett_luce(log_scores: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
    """
    Example:
    >>> log_scores = torch.tensor([0, 10., 20.])
    >>> sample_plackett_luce(log_scores)
    tensor([[2, 1, 0]])
    """

    X = torch.distributions.Exponential(torch.exp(log_scores)).sample((num_samples,))  # Exponential samples
    ranking = torch.argsort(X)  # Sort by smallest X
    return ranking


def log_plackett_luce_prob(log_scores: torch.Tensor, ranking: torch.Tensor) -> torch.Tensor:
    """
    log_scores: torch.Tensor, shape (n_nodes,)
    ranking: torch.Tensor, shape (b_size, n_nodes)

    Returns:
    log_prob: torch.Tensor, shape (b_size,)

    Example:
    >>> log_scores = torch.tensor([np.log(1), np.log(2), np.log(3)])
    >>> ranking = torch.tensor([[2, 1, 0]]).int()
    >>> log_plackett_luce_prob(log_scores, ranking)
    tensor([-1.0986])
    """
    permuted_log_scores = log_scores[ranking]  # shape: (b_size, n_nodes)
    rhs = torch.logcumsumexp(permuted_log_scores.flip(1), dim=1).sum(dim=1)  # shape: (b_size,)
    lhs = log_scores.sum(dim=0, keepdim=True)  # shape: (1,)
    return lhs - rhs  # shape: (b_size,)


class PlackettLuceTrainer:
    def __init__(
        self,
        model: OSlow,
        dag: nx.DiGraph,
        flow_dataloader: DataLoader,
        flow_optimizer: Callable[[Iterable], torch.optim.Optimizer],
        permutation_optimizer: Callable[[Iterable], torch.optim.Optimizer],
        perm_expectation_b_size: int,
        max_epochs: int,
        flow_lr_scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler],
        device: str = "cpu",
    ):
        self.device = device
        self.max_epochs = max_epochs
        self.model = model.to(device)

        self.flow_dataloader = flow_dataloader
        self.flow_optimizer_instantiate = flow_optimizer
        self.flow_lr_scheduler_instantiate = flow_lr_scheduler
        self.permutation_optimizer_instantiate = permutation_optimizer
        self.perm_expectation_b_size = perm_expectation_b_size

        self.permutation_log_scores = torch.nn.Parameter(
            torch.ones(len(dag.nodes), device=device)
        )  # All one initialize (TODO)

        # save the dag for evaluation
        self.dag = dag
        self.perm_step_count = 0
        self.flow_step_count = 0

    def run(self):
        self.model.train()
        self.model = self.model.to(self.device)

        self.flow_optimizer = self.flow_optimizer_instantiate(self.model.parameters())
        self.flow_scheduler = self.flow_lr_scheduler_instantiate(self.flow_optimizer)

        self.permutation_optimizer = self.permutation_optimizer_instantiate([self.permutation_log_scores])

        for epoch in tqdm(range(self.max_epochs)):
            avg_loss = []
            for batch in self.flow_dataloader:
                # Normalize the permutation_log_scores
                with torch.no_grad():
                    self.permutation_log_scores -= torch.logsumexp(self.permutation_log_scores, dim=0)

                print(self.permutation_log_scores)
                batch = batch.to(self.model.device)
                b_size = batch.shape[0]
                n_nodes = batch.shape[1]

                # sample from plackett luce with shape (b_size, perm_expectation_b_size, n_nodes)
                perm_vector = sample_plackett_luce(self.permutation_log_scores, b_size * self.perm_expectation_b_size)
                perm_matrices = listperm2matperm(perm_vector, device=self.model.device)

                # shape: (b_size, perm_expectation_b_size, n_nodes)
                batch_repeated = batch.unsqueeze(1).repeat(
                    1, self.perm_expectation_b_size, 1
                )  # shape: (b_size, perm_expectation_b_size, n_nodes)

                # train flow and permutation together
                self.flow_optimizer.zero_grad()
                self.permutation_optimizer.zero_grad()

                log_probs = self.model.log_prob(
                    batch_repeated.reshape(-1, n_nodes), perm_mat=perm_matrices.float()
                ).reshape(
                    b_size, self.perm_expectation_b_size, 1
                )  # f_\theta(X, \sigma) with shape (b_size, perm_expectation_b_size, 1)

                flow_loss = -log_probs.exp().mean(dim=1).log().mean()

                log_perm_probs = log_plackett_luce_prob(self.permutation_log_scores, perm_vector).reshape(
                    b_size, self.perm_expectation_b_size, 1
                )  # shape: (b_size, perm_expectation_b_size, 1)

                perm_loss = -(
                    (log_perm_probs * log_probs.detach().exp()).mean(dim=1) / (log_probs.detach().exp()).mean(dim=1)
                ).mean()

                total_loss = perm_loss + flow_loss

                total_loss.backward()
                self.flow_optimizer.step()
                self.permutation_optimizer.step()
                self.flow_step_count += 1
                self.perm_step_count += 1

                wandb.log({f"step": self.flow_step_count})
                wandb.log({f"total_loss": total_loss.item()})
                wandb.log({f"perm_loss": perm_loss.item()})
                wandb.log({f"flow_loss": flow_loss.item()})
                avg_loss.append(total_loss.item())

            # Perform a learning rate scheduling step
            if isinstance(self.flow_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.flow_scheduler.step(sum(avg_loss) / len(avg_loss))
            else:
                self.flow_scheduler.step()

            learned_perm = torch.argsort(-self.permutation_log_scores)
            cbc_metric = backward_relative_penalty(perm=learned_perm.tolist(), dag=self.dag)
            wandb.log({"learned_perm": str(learned_perm.tolist())})
            wandb.log({"learned_perm_cbc_metric": cbc_metric})

            wandb.log({"epoch": epoch})
