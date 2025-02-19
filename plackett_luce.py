import hydra
import wandb
import os
import torch

import networkx as nx

from abc import ABC, abstractmethod
from omegaconf import OmegaConf
from pprint import pprint
from random_word import RandomWords
from typing import Callable, Iterable, Literal, List
from tqdm import tqdm

from oslow.models.oslow import OSlow
from oslow.training.utils import listperm2matperm, seed_everything
from oslow.evaluation import backward_relative_penalty
from oslow.data import OCDDataset


class MultiTaskCoeffStrategy(ABC):
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks

    @abstractmethod
    def __call__(self, iter: int, *args, **kwargs) -> List[float]:
        pass


class DWAMultiTaskStrategy(MultiTaskCoeffStrategy):
    """
    Dynamic Weight Average (DWA) from https://arxiv.org/pdf/1803.10704
    """

    def __init__(self, num_tasks: int, temperature: float):
        super().__init__(num_tasks)
        self.temperature = temperature
        self.loss_t_minus_1 = None
        self.loss_t_minus_2 = None

    def __call__(self, iter: int, losses: torch.Tensor | None = None):
        if losses is None:
            return [1.0 for _ in range(self.num_tasks)]

        if self.loss_t_minus_1 is None:
            self.loss_t_minus_1 = losses.detach()

        if self.loss_t_minus_2 is None:
            self.loss_t_minus_2 = losses.detach()

        weights = torch.exp(self.loss_t_minus_1) / torch.exp(self.loss_t_minus_2)
        self.loss_t_minus_2 = self.loss_t_minus_1
        self.loss_t_minus_1 = losses.detach()
        exp_weights_temp = torch.exp(weights / self.temperature)

        min_val = torch.nan_to_num(exp_weights_temp, nan=float("inf")).min()
        exp_weights_temp = exp_weights_temp.nan_to_num(min_val)
        lambdas = exp_weights_temp / exp_weights_temp.sum() * self.num_tasks

        return lambdas.tolist()


class ConstantStrategy(MultiTaskCoeffStrategy):
    def __init__(self, num_tasks: int, value: float):
        super().__init__(num_tasks=num_tasks)
        self.value = value

    def __call__(self, iter: int, *args, **kwargs) -> List[float]:
        return [self.value for _ in range(self.num_tasks)]


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
        optimizer: Callable[[Iterable], torch.optim.Optimizer],
        batch_size: int,
        perm_expectation_b_size: int,
        max_epochs: int,
        lr_scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler],
        device: str,
        reinforce_baseline: Literal["mean", "zero"],
        flow_learning_iters: int,
        perm_learning_iters: int,
        perm_optimizer: Callable[[Iterable], torch.optim.Optimizer] | None,
        perm_lr_scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler] | None,
        normalize_scores: bool,
        sampling: Literal["exponential_race", "gumbel"],
        coeff_scheduling_strategy: MultiTaskCoeffStrategy | None,
        restart_flow: bool,
    ):
        self.device = device
        self.max_epochs = max_epochs
        self.model = model.to(device)

        self.dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        self.flow_optimizer_instantiate = optimizer
        self.flow_lr_scheduler_instantiate = lr_scheduler

        self.perm_expectation_b_size = perm_expectation_b_size

        self.perm_optimizer_instantiate = perm_optimizer or optimizer
        self.perm_lr_scheduler_instantiate = perm_lr_scheduler or lr_scheduler

        # save the dag for evaluation
        self.dag = data.dag

        # All one initialize (TODO)
        self.permutation_log_scores = torch.nn.Parameter(torch.zeros(len(self.dag.nodes), device=device))

        self.step_count = 0

        if coeff_scheduling_strategy is None:
            coeff_scheduling_strategy = ConstantStrategy(num_tasks=2, value=1.0)
        else:
            self.multi_task_scheduler = coeff_scheduling_strategy

        self.reinforce_baseline = reinforce_baseline

        self.alternating_training = flow_learning_iters > 0 and perm_learning_iters > 0
        self.flow_learning_iters = flow_learning_iters
        self.perm_learning_iters = perm_learning_iters

        self.normalize_scores = normalize_scores

        if sampling == "exponential_race":
            self.sample_fn = sample_plackett_luce
        else:
            raise ValueError(f"Sampling method {sampling} not supported!")

        self.restart_flow = restart_flow

        # Log the correct order
        wandb.log({"score/correct_order": str(list(nx.topological_sort(self.dag)))}, step=self.step_count)

    def run(self):
        self.model.train()
        self.model = self.model.to(self.device)

        flow_optimizer = self.flow_optimizer_instantiate(self.model.parameters())
        flow_lr_scheduler = self.flow_lr_scheduler_instantiate(flow_optimizer)

        perm_optimizer = self.perm_optimizer_instantiate([self.permutation_log_scores])
        perm_lr_scheduler = self.perm_lr_scheduler_instantiate(perm_optimizer)

        perm_coeff, flow_coeff = 1.0, 1.0

        if self.alternating_training:
            flow_training, perm_training = True, False
            print("Alternating training")
        else:
            flow_training, perm_training = True, True
            print("Joint training")

        flow_step, perm_step = 0, 0

        for epoch in tqdm(range(self.max_epochs)):
            flow_avg_loss = []
            perm_avg_loss = []
            for batch in self.dataloader:
                batch: torch.Tensor

                # Normalize the permutation_log_scores
                if self.normalize_scores:
                    with torch.no_grad():
                        self.permutation_log_scores -= torch.logsumexp(self.permutation_log_scores, dim=0)

                batch = batch.to(self.model.device)
                b_size = batch.shape[0]
                n_nodes = batch.shape[1]

                # sample from plackett luce with shape (b_size, perm_expectation_b_size, n_nodes)
                perm_vector = self.sample_fn(self.permutation_log_scores, b_size * self.perm_expectation_b_size)
                perm_matrices = listperm2matperm(perm_vector, device=self.model.device)

                # shape: (b_size, perm_expectation_b_size, n_nodes)
                batch_repeated = batch.unsqueeze(1).repeat(1, self.perm_expectation_b_size, 1)

                # shape (b_size, perm_expectation_b_size, 1)
                log_probs = self.model.log_prob(
                    batch_repeated.reshape(-1, n_nodes), perm_mat=perm_matrices.float()
                ).reshape(b_size, self.perm_expectation_b_size, 1)

                denom = (log_probs.detach().exp()).mean(dim=1, keepdim=True) + 1e-6
                numerator = log_probs.detach().exp()
                weights = numerator / denom

                if flow_training:
                    flow_loss = -((log_probs * weights).mean(dim=1)).mean()
                    flow_step += 1
                    if self.alternating_training and flow_step >= self.flow_learning_iters:
                        flow_training, perm_training = False, True
                        flow_step = 0

                    wandb.log({f"loss/flow_loss": flow_loss.item()}, step=self.step_count)
                    flow_avg_loss.append(flow_loss.item())

                if perm_training:
                    if self.reinforce_baseline == "mean":
                        baseline = weights.mean()
                    elif self.reinforce_baseline == "zero":
                        baseline = 0.0
                    else:
                        raise ValueError(f"Baseline {self.reinforce_baseline} not supported!")

                    # shape: (b_size, perm_expectation_b_size, 1)
                    log_perm_probs = log_plackett_luce_prob(self.permutation_log_scores, perm_vector).reshape(
                        b_size, self.perm_expectation_b_size, 1
                    )

                    perm_loss = -((log_perm_probs * (weights - baseline)).mean(dim=1)).mean()
                    perm_step += 1
                    if self.alternating_training and perm_step >= self.perm_learning_iters:
                        flow_training, perm_training = True, False
                        perm_step = 0
                        if self.restart_flow:
                            self.model.reinitialize()
                            flow_optimizer = self.flow_optimizer_instantiate(self.model.parameters())
                            flow_lr_scheduler = self.flow_lr_scheduler_instantiate(flow_optimizer)

                    wandb.log({f"loss/perm_loss": perm_loss.item()}, step=self.step_count)
                    perm_avg_loss.append(perm_loss.item())

                if flow_training and perm_training:
                    total_loss = flow_coeff * flow_loss + perm_coeff * perm_loss
                    flow_optimizer.zero_grad()
                    perm_optimizer.zero_grad()
                    total_loss.backward()
                    flow_optimizer.step()
                    perm_optimizer.step()

                    wandb.log({f"loss/total_loss": total_loss.item()}, step=self.step_count)

                    perm_coeff, flow_coeff = self.multi_task_scheduler(
                        self.step_count, losses=torch.tensor([perm_loss, flow_loss])
                    )

                    wandb.log({"coeffs/perm_coeff": perm_coeff}, step=self.step_count)
                    wandb.log({"coeffs/flow_coeff": flow_coeff}, step=self.step_count)
                elif flow_training:
                    flow_optimizer.zero_grad()
                    flow_loss.backward()
                    flow_optimizer.step()
                elif perm_training:
                    perm_optimizer.zero_grad()
                    perm_loss.backward()
                    perm_optimizer.step()

                if flow_training:
                    if isinstance(flow_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        flow_lr_scheduler.step(sum(flow_avg_loss) / len(flow_avg_loss))
                    else:
                        flow_lr_scheduler.step()

                    wandb.log({"flow_lr": flow_optimizer.param_groups[0]["lr"]}, step=self.step_count)

                if perm_training:
                    if isinstance(perm_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        perm_lr_scheduler.step(sum(perm_avg_loss) / len(perm_avg_loss))
                    else:
                        perm_lr_scheduler.step()
                    wandb.log({"perm_lr": perm_optimizer.param_groups[0]["lr"]}, step=self.step_count)

                self.step_count += 1
                scores = {}
                for i in range(len(self.permutation_log_scores)):
                    scores[f"score/{i}"] = self.permutation_log_scores[i].item()

                wandb.log(scores, step=self.step_count)

                wandb.log({f"step": self.step_count}, step=self.step_count)

            learned_perm = torch.argsort(-self.permutation_log_scores)
            cbc_metric = backward_relative_penalty(perm=learned_perm.tolist(), dag=self.dag)
            wandb.log({"eval/learned_perm_cbc_metric": cbc_metric}, step=self.step_count)

            wandb.log({"epoch": epoch}, step=self.step_count)


def get_torch_distribution(distr_name):
    if distr_name == "laplace":
        return "torch.distributions.Laplace"
    elif distr_name == "normal":
        return "torch.distributions.Normal"
    elif distr_name == "uniform":
        return "torch.distributions.Uniform"
    else:
        ValueError(f"Distribution {distr_name} cannot be resolved!")


def get_torch_distribution_args(distr_name):
    if distr_name == "laplace":
        return [0.0, 1.0]
    elif distr_name == "normal":
        return [0.0, 1.0]
    elif distr_name == "uniform":
        return [0.0, 1.0]
    else:
        ValueError(f"Distribution {distr_name} cannot be resolved!")


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


# TODO make the run faster by not having different permutations for each sample


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

        trainer = PlackettLuceTrainer(
            model=conf.model,
            data=conf.data,
            optimizer=conf.optimizer,
            batch_size=conf.batch_size,
            perm_expectation_b_size=conf.perm_expectation_b_size,
            max_epochs=conf.max_epochs,
            lr_scheduler=conf.lr_scheduler,
            device=conf.device,
            reinforce_baseline=conf.reinforce_baseline,
            flow_learning_iters=conf.flow_learning_iters,
            perm_learning_iters=conf.perm_learning_iters,
            perm_optimizer=conf.perm_optimizer,
            perm_lr_scheduler=conf.perm_lr_scheduler,
            normalize_scores=conf.normalize_scores,
            sampling=conf.sampling,
            coeff_scheduling_strategy=conf.coeff_scheduling_strategy,
            restart_flow=conf.restart_flow,
        )
        trainer.run()


if __name__ == "__main__":
    main()
