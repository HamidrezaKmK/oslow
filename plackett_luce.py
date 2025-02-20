import hydra
import wandb
import os
import torch

import networkx as nx

from omegaconf import OmegaConf
from pprint import pprint
from random_word import RandomWords
from typing import Callable, Iterable, Literal
from tqdm import tqdm

from oslow.models.oslow import OSlow
from oslow.training.utils import listperm2matperm, seed_everything
from oslow.evaluation import backward_relative_penalty
from oslow.data import OCDDataset


# TODO: Plackett-Luce with Reinforce + Gumbel-Max for sampling
# TODO: Plackett-Luce with exponential race sampling but soft-sort gradient straight-through estimation
# TODO: No Plackett-Luce and just Gubmel-Sinkhorn
# TODO: Read Ermon's paper on Plackett-Luce
# TODO: Read https://arxiv.org/pdf/2006.16038
# TODO: Hyper-parameter Sweep 


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
        data: OCDDataset,
        flow_optimizer: Callable[[Iterable], torch.optim.Optimizer],
        flow_batch_size: int,
        permutation_batch_size: int,
        perm_expectation_b_size: int,
        rounds: int,
        flow_lr_scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler],
        device: str,
        reinforce_baseline: Literal["mean", "zero"],
        flow_learning_epochs: int,
        perm_learning_epochs: int,
        perm_optimizer: Callable[[Iterable], torch.optim.Optimizer],
        perm_lr_scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler],
        normalize_scores: bool,
        sampling: Literal["exponential_race", "gumbel"],
        restart_flow: bool,
    ):
        self.device = device
        self.rounds = rounds
        self.model = model.to(device)

        self.flow_dataloader = torch.utils.data.DataLoader(data, batch_size=flow_batch_size, shuffle=True)
        self.perm_dataloader = torch.utils.data.DataLoader(data, batch_size=permutation_batch_size, shuffle=True)
        self.flow_optimizer_instantiate = flow_optimizer
        self.flow_lr_scheduler_instantiate = flow_lr_scheduler

        self.perm_expectation_b_size = perm_expectation_b_size

        self.perm_optimizer_instantiate = perm_optimizer
        self.perm_lr_scheduler_instantiate = perm_lr_scheduler

        # save the dag for evaluation
        self.dag = data.dag

        # All one initialize (TODO)
        self.permutation_log_scores = torch.nn.Parameter(torch.zeros(len(self.dag.nodes), device=device))

        self.flow_step_count = 0
        self.perm_step_count = 0

        self.reinforce_baseline = reinforce_baseline

        self.flow_learning_epochs = flow_learning_epochs
        self.perm_learning_epochs = perm_learning_epochs

        self.normalize_scores = normalize_scores

        if sampling == "exponential_race":
            self.sample_fn = sample_plackett_luce
        else:
            raise ValueError(f"Sampling method {sampling} not supported!")

        self.restart_flow = restart_flow

        # Log the correct order
        wandb.log({"permutation/correct_order": str(list(nx.topological_sort(self.dag))), "permutation/step": 0})

    def run(self):
        self.model.train()
        self.model = self.model.to(self.device)

        flow_optimizer = self.flow_optimizer_instantiate(self.model.parameters())
        flow_lr_scheduler = self.flow_lr_scheduler_instantiate(flow_optimizer)

        perm_optimizer = self.perm_optimizer_instantiate([self.permutation_log_scores])
        perm_lr_scheduler = self.perm_lr_scheduler_instantiate(perm_optimizer)

        for round_ in tqdm(range(self.rounds), desc="Round"):
            # Normalize the permutation_log_scores
            if self.normalize_scores:
                with torch.no_grad():
                    self.permutation_log_scores -= torch.logsumexp(self.permutation_log_scores, dim=0)

            for flow_epoch in tqdm(range(self.flow_learning_epochs), desc="Flow"):
                flow_avg_loss = [0.0] * len(self.flow_dataloader)
                for i, batch in enumerate(self.flow_dataloader):
                    batch: torch.Tensor
                    batch = batch.to(self.device)
                    b_size = batch.shape[0]

                    perm_matrices = listperm2matperm(
                        self.sample_fn(self.permutation_log_scores, b_size), device=self.device
                    )  # shape: (b_size, n_nodes, n_nodes)

                    flow_optimizer.zero_grad()

                    log_probs = self.model.log_prob(batch, perm_mat=perm_matrices.float())  # shape (b_size, 1)
                    flow_loss = -log_probs.mean()

                    flow_loss.backward()
                    flow_optimizer.step()

                    wandb.log({"flow/step": self.flow_step_count, "flow/loss": flow_loss.item()})
                    flow_avg_loss[i] = flow_loss.item()

                    self.flow_step_count += 1

                if isinstance(flow_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    flow_lr_scheduler.step(sum(flow_avg_loss) / len(flow_avg_loss))
                else:
                    flow_lr_scheduler.step()

                wandb.log(
                    {
                        "flow/lr": flow_optimizer.param_groups[0]["lr"],
                        "flow/epoch": flow_epoch,
                        "flow/step": self.flow_step_count,
                    }
                )

            for perm_epoch in tqdm(range(self.perm_learning_epochs), desc="Permutation"):
                perm_avg_loss = [0.0] * len(self.perm_dataloader)
                for i, batch in enumerate(self.perm_dataloader):
                    batch: torch.Tensor
                    batch = batch.to(self.device)
                    b_size = batch.shape[0]

                    # sample from plackett luce with shape (b_size * perm_expectation_b_size, n_nodes)
                    perm_vector = self.sample_fn(self.permutation_log_scores, b_size * self.perm_expectation_b_size)
                    perm_matrices = listperm2matperm(perm_vector, device=self.device).float()

                    # shape: (b_size * perm_expectation_b_size, n_nodes)
                    batch_repeated = batch.repeat(self.perm_expectation_b_size, 1)

                    # shape: (b_size, perm_expectation_b_size, 1)
                    log_probs = self.model.log_prob(batch_repeated, perm_mat=perm_matrices).reshape(b_size, -1, 1)

                    numerator = log_probs.detach().exp()
                    # TODO can be estimated using a different set of samples (for sample size = 1)
                    denom = numerator.mean(dim=1, keepdim=True) + 1e-6  # shape: (b_size, 1, 1)
                    weights = numerator / denom

                    if self.reinforce_baseline == "mean":
                        baseline = weights.mean()
                    elif self.reinforce_baseline == "zero":
                        baseline = 0.0
                    else:
                        raise ValueError(f"Baseline {self.reinforce_baseline} not supported!")

                    log_perm_probs = log_plackett_luce_prob(self.permutation_log_scores, perm_vector).reshape(
                        b_size, -1, 1
                    )  # shape: (b_size, perm_expectation_b_size, 1)

                    perm_optimizer.zero_grad()

                    perm_loss = -((log_perm_probs * (weights - baseline)).mean(dim=1)).mean()

                    perm_loss.backward()
                    perm_optimizer.step()

                    perm_avg_loss[i] = perm_loss.item()

                    self.perm_step_count += 1

                    scores = {
                        f"permutation/score/{i}": self.permutation_log_scores[i].item()
                        for i in range(len(self.permutation_log_scores))
                    }

                    metrics = {
                        "permutation/step": self.perm_step_count,
                        "permutation/loss": perm_loss.item(),
                        **scores,
                    }
                    wandb.log(metrics)

                if isinstance(perm_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    perm_lr_scheduler.step(sum(perm_avg_loss) / len(perm_avg_loss))
                else:
                    perm_lr_scheduler.step()

                learned_perm = torch.argsort(-self.permutation_log_scores)
                cbc_metric = backward_relative_penalty(perm=learned_perm.tolist(), dag=self.dag)

                metrics = {
                    "permutation/lr": perm_optimizer.param_groups[0]["lr"],
                    "permutation/epoch": perm_epoch,
                    "permutation/learned_perm_cbc_metric": cbc_metric,
                    "permutation/step": self.perm_step_count,
                }

                wandb.log(metrics)

            if self.restart_flow:
                self.model.reinitialize()
                flow_optimizer = self.flow_optimizer_instantiate(self.model.parameters())
                flow_lr_scheduler = self.flow_lr_scheduler_instantiate(flow_optimizer)

            wandb.log({"round": round_})


def get_torch_distribution(distr_name):
    if distr_name == "laplace":
        return "torch.distributions.Laplace"
    elif distr_name == "normal":
        return "torch.distributions.Normal"
    elif distr_name == "uniform":
        return "torch.distributions.Uniform"
    else:
        raise ValueError(f"Distribution {distr_name} cannot be resolved!")


def get_torch_distribution_args(distr_name):
    if distr_name == "laplace":
        return [0.0, 1.0]
    elif distr_name == "normal":
        return [0.0, 1.0]
    elif distr_name == "uniform":
        return [0.0, 1.0]
    else:
        raise ValueError(f"Distribution {distr_name} cannot be resolved!")


# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("get_torch_distribution", get_torch_distribution)
OmegaConf.register_new_resolver("get_torch_distribution_args", get_torch_distribution_args)


def init_run_dir(conf, base_name=None):
    # Handle preemption and resume
    run_name = str(conf.wandb.run_name)
    resume = True
    r = RandomWords()
    w1, w2 = r.get_random_word(), r.get_random_word()
    if conf.wandb.run_name is not None:
        run_name = str(conf.wandb.run_name) + base_name
    else:
        run_name = base_name

    run_name += f"_{w1}_{w2}"

    out_dir = os.path.join(conf.out_dir, run_name)

    config_yaml = os.path.join(out_dir, "config.yaml")
    if os.path.exists(config_yaml):
        with open(config_yaml) as fp:
            old_conf = OmegaConf.load(fp.name)
        run_id = old_conf.wandb.run_id
    else:
        run_id = wandb.util.generate_id()
        resume = False

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        resume = False

    conf.out_dir = out_dir
    conf.wandb.resume = resume
    conf.wandb.run_id = run_id
    conf.wandb.run_name = run_name
    return conf


@hydra.main(version_base=None, config_path="config", config_name="plackett_luce")
def main(conf):
    seed_everything(conf.seed)
    model_type = "additive" if conf.data.additive else "affine"
    num_nodes = conf.data.graph_generator.num_nodes
    graph_type = conf.data.graph_generator.graph_type
    noise_type = conf.data.noise_generator.noise_type
    if "link" in conf.data:
        link_function = conf.data.link
        run_name = f"{link_function}_{noise_type}_{model_type}_{graph_type}_d{num_nodes}"
    else:
        run_name = f"nonparametric_{noise_type}_{model_type}_{graph_type}_d{num_nodes}"

    conf = hydra.utils.instantiate(conf)
    if conf.test_run:
        pprint(OmegaConf.to_container(conf, resolve=True))
    else:
        conf = init_run_dir(conf, run_name)
        wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project,
            config=OmegaConf.to_container(conf, resolve=True),
            name=conf.wandb.run_name,
            id=conf.wandb.run_id,
            resume="allow" if conf.wandb.resume else False,
            # compatible with hydra
            settings=wandb.Settings(start_method="thread"),
        )

        wandb.define_metric("flow/step")
        wandb.define_metric("permutation/step")
        wandb.define_metric("flow/*", step_metric="flow/step")
        wandb.define_metric("permutation/*", step_metric="permutation/step")

        trainer = PlackettLuceTrainer(
            model=conf.model,
            data=conf.data,
            flow_optimizer=conf.flow_optimizer,
            flow_batch_size=conf.flow_batch_size,
            permutation_batch_size=conf.permutation_batch_size,
            perm_expectation_b_size=conf.perm_expectation_b_size,
            rounds=conf.rounds,
            flow_lr_scheduler=conf.flow_lr_scheduler,
            device=conf.device,
            reinforce_baseline=conf.reinforce_baseline,
            flow_learning_epochs=conf.flow_learning_epochs,
            perm_learning_epochs=conf.perm_learning_epochs,
            perm_optimizer=conf.perm_optimizer,
            perm_lr_scheduler=conf.perm_lr_scheduler,
            normalize_scores=conf.normalize_scores,
            sampling=conf.sampling,
            restart_flow=conf.restart_flow,
        )
        trainer.run()


if __name__ == "__main__":
    main()
