import hydra
import wandb
import os
import torch

import networkx as nx
import pandas as pd
import numpy as np

from omegaconf import OmegaConf
from pprint import pprint
from random_word import RandomWords
from typing import Callable, Iterable, Literal, List, Dict
from tqdm import tqdm

from oslow.models.oslow import OSlow
from oslow.training.utils import listperm2matperm, seed_everything
from oslow.evaluation import backward_relative_penalty, sid, shd
from oslow.data import OCDDataset

from oslow.post_processing.cam_pruning import sparse_regression_based_pruning
from oslow.post_processing.pc_pruning import pc_based_pruning
from oslow.post_processing.ultimate_pruning import ultimate_pruning

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.SkeletonDiscovery import skeleton_discovery
from causallearn.utils.cit import kci, gsq, FisherZ, CIT

# TODO compare the best permutation from the training to the actual best permutation at the end
# TODO Learn the connected component first and then do flow training on it (Placett-Luce on partially ordered lists)

# TODO: Plackett-Luce with exponential race / Gumbel-Max sampling but soft-sort gradient straight-through estimation
# TODO: No Plackett-Luce and just Gubmel-Sinkhorn
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
        self._permutation_log_scores = torch.nn.Parameter(torch.zeros(len(self.dag.nodes), device=device))

        self.flow_step_count = 0
        self.perm_step_count = 0

        self.reinforce_baseline = reinforce_baseline

        self.flow_learning_epochs = flow_learning_epochs
        self.perm_learning_epochs = perm_learning_epochs

        self.normalize_scores = normalize_scores

        self.restart_flow = restart_flow

        # Log the correct order
        wandb.log({"permutation/correct_order": str(list(nx.topological_sort(self.dag))), "permutation/step": 0})

    @property
    def permutation_log_scores(self) -> torch.Tensor:
        if self.normalize_scores:
            with torch.no_grad():
                self._permutation_log_scores -= torch.logsumexp(self._permutation_log_scores, dim=0)

        return self._permutation_log_scores

    def run(self) -> List[int]:
        self.model.train()
        self.model = self.model.to(self.device)

        flow_optimizer = self.flow_optimizer_instantiate(self.model.parameters())
        flow_lr_scheduler = self.flow_lr_scheduler_instantiate(flow_optimizer)

        perm_optimizer = self.perm_optimizer_instantiate([self.permutation_log_scores])
        perm_lr_scheduler = self.perm_lr_scheduler_instantiate(perm_optimizer)

        learned_perm = None

        for round_ in tqdm(range(self.rounds), desc="Round"):
            self.model.train()
            
            for param in self.model.parameters():
                param.requires_grad = True
                
            for flow_epoch in tqdm(range(self.flow_learning_epochs), desc="Flow"):
                flow_avg_loss = [0.0] * len(self.flow_dataloader)
                for i, batch in enumerate(self.flow_dataloader):
                    batch: torch.Tensor
                    batch = batch.to(self.device)
                    b_size = batch.shape[0]

                    perm_matrices = listperm2matperm(
                        sample_plackett_luce(self.permutation_log_scores, b_size), device=self.device
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

            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            for perm_epoch in tqdm(range(self.perm_learning_epochs), desc="Permutation"):
                perm_avg_loss = [0.0] * len(self.perm_dataloader)
                for i, batch in enumerate(self.perm_dataloader):
                    batch: torch.Tensor
                    batch = batch.to(self.device)
                    b_size = batch.shape[0]

                    # sample from plackett luce with shape (b_size * perm_expectation_b_size, n_nodes)
                    perm_vector = sample_plackett_luce(
                        self.permutation_log_scores, b_size * self.perm_expectation_b_size
                    )
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

        return learned_perm.tolist()


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
    base_name = "" if base_name is None else base_name
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
    conf.wandb.run_id = run_id
    conf.wandb.run_name = run_name
    return conf


def metrics_fn(
    order: List[int], samples: pd.DataFrame, true_dag: nx.DiGraph, method: Literal["pc", "cam", "ultimate"] = "pc"
) -> Dict[str, float]:
    if method == "pc":
        dag = pc_based_pruning(samples, order, verbose=False)
    elif method == "cam":
        dag = sparse_regression_based_pruning(samples, order)
    elif method == "ultimate":
        dag = ultimate_pruning(samples, order)
    else:
        raise NotImplementedError()
    return {"SID": sid(true_dag, dag), "SHD": shd(true_dag, dag), "CBC": backward_relative_penalty(order, true_dag, normalize=True)}


@hydra.main(version_base=None, config_path="config", config_name="plackett_luce")
def main(conf):
    seed_everything(conf.seed)
    run_name = None
    if "additive" in conf.data and "graph_generator" in conf.data and "noise_generator" in conf.data:
        model_type = "additive" if conf.data.additive else "affine"
        num_nodes = conf.data.graph_generator.num_nodes
        graph_type = conf.data.graph_generator.graph_type
        noise_type = conf.data.noise_generator.noise_type
        pnl_transform = conf.data.post_non_linear_transform
        if "link" in conf.data:
            link_function = conf.data.link
            if pnl_transform is not None:
                run_name = f"pnl_{pnl_transform}-{link_function}_{noise_type}_{model_type}_{graph_type}_d{num_nodes}"
            else:
                run_name = f"{link_function}_{noise_type}_{model_type}_{graph_type}_d{num_nodes}"
        else:
            run_name = f"nonparametric_{noise_type}_{model_type}_{graph_type}_d{num_nodes}"
    
    instantiated = hydra.utils.instantiate(conf)
    instantiated = init_run_dir(instantiated, run_name)
    wandb.init(
        dir=instantiated.out_dir,
        project=instantiated.wandb.project,
        config=OmegaConf.to_container(instantiated, resolve=True),
        name=instantiated.wandb.run_name,
        id=instantiated.wandb.run_id,
        # compatible with hydra
        settings=wandb.Settings(start_method="thread"),
    )
    if conf.get("base_graph", None) == "pc":
        all_values = instantiated.data.samples.values
        cg = skeleton_discovery(all_values, alpha=0.1, indep_test=CIT(all_values, method='fisherz'))
        # cg = pc(all_values, 0.001, kci) # , kci
        base_graph = cg.G.graph
    else: # assume it is a numpy array
        base_graph = np.eye(len(instantiated.data.samples.columns), dtype=int) - 1
    seen = set()

    def dfs(graph: np.ndarray, u, columns: List):
        seen.add(columns.get_loc(u))
        for v in range(graph.shape[0]):
            if graph[u, v] != 0 and columns.get_loc(v) not in seen:
                seen.add(columns.get_loc(v))
                dfs(graph, v, columns)

    perm_learned = []
    for i, v in enumerate(instantiated.data.samples.columns):
        if v in seen:
            continue
        seen_before = seen.copy()
        dfs(base_graph, i, instantiated.data.samples.columns)
        connected_component = list(seen - seen_before)
        conf.model.in_features = len(connected_component)
        conf_instantiated = hydra.utils.instantiate(conf)
        mapping = {connected_component[j]: j for j in range(len(connected_component))}
        mapping_rev = {j: connected_component[j] for j in range(len(connected_component))}
        sub_df = instantiated.data.samples[connected_component]
        sub_df = sub_df.rename(columns=mapping)
        sub_graph = nx.subgraph(instantiated.data.dag, connected_component)
        sub_graph = nx.relabel_nodes(sub_graph, mapping, copy=True)
        data = OCDDataset(
            samples=sub_df,
            dag=sub_graph,
        )
        trainer = PlackettLuceTrainer(
            model=conf_instantiated.model,
            data=data,
            flow_optimizer=conf_instantiated.flow_optimizer,
            flow_batch_size=conf_instantiated.flow_batch_size,
            permutation_batch_size=conf_instantiated.permutation_batch_size,
            perm_expectation_b_size=conf_instantiated.perm_expectation_b_size,
            rounds=conf_instantiated.rounds,
            flow_lr_scheduler=conf_instantiated.flow_lr_scheduler,
            device=conf_instantiated.device,
            reinforce_baseline=conf_instantiated.reinforce_baseline,
            flow_learning_epochs=conf_instantiated.flow_learning_epochs,
            perm_learning_epochs=conf_instantiated.perm_learning_epochs,
            perm_optimizer=conf_instantiated.perm_optimizer,
            perm_lr_scheduler=conf_instantiated.perm_lr_scheduler,
            normalize_scores=conf_instantiated.normalize_scores,
            restart_flow=conf_instantiated.restart_flow,
        )

        perm_learned_ = trainer.run()
        perm_learned_ = [mapping_rev[i] for i in perm_learned_]
        perm_learned.extend(perm_learned_)

    if instantiated.post_processing_method is not None:     
        metrics = metrics_fn(perm_learned, instantiated.data.samples, instantiated.data.dag, method=instantiated.post_processing_method)
        wandb.log(metrics)
    wandb.finish()


if __name__ == "__main__":
    main()
