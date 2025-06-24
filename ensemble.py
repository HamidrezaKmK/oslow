import hydra
import wandb
import os
import itertools
import torch
import random

from omegaconf import OmegaConf
from pprint import pprint
from random_word import RandomWords
from typing import List

from oslow.training.utils import seed_everything


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


def get_permutations(n: int) -> List[List[int]]:
    return list(itertools.permutations(range(n)))


def get_random_permutations_subset(n: int, k: int, seed: int = 96875) -> List[List[int]]:
    random.seed(seed)
    permutations_set = set()
    numbers = list(range(n))

    while len(permutations_set) < k:
        random.shuffle(numbers)
        permutations_set.add(tuple(numbers))

    return list(permutations_set)


# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("get_torch_distribution", get_torch_distribution)
OmegaConf.register_new_resolver("get_torch_distribution_args", get_torch_distribution_args)
OmegaConf.register_new_resolver("get_permutations", get_permutations)
OmegaConf.register_new_resolver("get_random_permutations_subset", get_random_permutations_subset)


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
    conf.wandb.resume = resume
    conf.wandb.run_id = run_id
    conf.wandb.run_name = run_name
    return conf


@hydra.main(version_base=None, config_path="config", config_name="ensemble")
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
                run_name = f"pnl_{pnl_transform}-{link_function}_{noise_type}_{graph_type}_d{num_nodes}"
            else:
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
        dset = conf.data
        flow_dloader = torch.utils.data.DataLoader(dset, batch_size=conf.flow_batch_size, shuffle=True)
        flow_ensemble_dloader = torch.utils.data.DataLoader(
            dset, batch_size=conf.flow_ensemble_batch_size, shuffle=True
        )
        model = conf.model

        trainer = conf.trainer(
            model=model,
            dag=dset.dag,
            flow_dataloader=flow_dloader,
            flow_ensemble_dataloader=flow_ensemble_dloader,
        )
        trainer.run()
        wandb.finish()

if __name__ == "__main__":
    main()
