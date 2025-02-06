import hydra
import wandb
import os

from omegaconf import OmegaConf
from pprint import pprint
from random_word import RandomWords
import networkx as nx
import torch
import numpy as np
import random as random


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
OmegaConf.register_new_resolver(
    "get_torch_distribution_args", get_torch_distribution_args
)

def save_synthetic_dataset(dataset, conf, run_name):
    """Save synthetic dataset and its graph structure."""
    # Create directory for synthetic datasets if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "synthetic_datasets")
    os.makedirs(data_dir, exist_ok=True)
    
    dataset.samples.to_csv(os.path.join(data_dir, f"{run_name}.csv"), index=False)
    
    # Save graph structure as edge list
    nx.write_edgelist(dataset.dag, os.path.join(data_dir, f"{run_name}.edgelist"))
    
    # Save configuration
    with open(os.path.join(data_dir, f"{run_name}.yaml"), "w") as f:
        OmegaConf.save(config=conf, f=f)

def init_run_dir(conf, base_name):
    # Handle preemption and resume
    resume = True
    seed = conf.seed
    if seed is None:
        seed = random.randint
    # r = RandomWords()
    # w1, w2 = r.get_random_word(), r.get_random_word()
    if conf.wandb.run_name is not None:
        run_name = f"{conf.wandb.run_name}_{base_name}_seed{seed}"
    else:
        run_name = f"{base_name}_seed{seed}"
    # run_name += f"_{w1}_{w2}"
    
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


@hydra.main(version_base=None, config_path="config", config_name="causal_discovery")
def main(conf):
    # a more informative run name with model type and data information
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
    
    if conf.get("seed") is None:
        # Generate random seed between 0 and 2^32-1
        conf.seed = torch.randint(0, 2**16-1, (1,)).item()
    
    seed = conf.seed
    # run_name = conf.run_name
    # Set all seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if conf.test_run:
        pprint(OmegaConf.to_container(conf, resolve=True))
    else:
        conf = init_run_dir(conf, run_name)
        wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project,
            config=OmegaConf.to_container(conf, resolve=True),
            name=str(seed)+conf.wandb.run_name,
            id=str(seed),
            resume="never",
            # compatible with hydra
            settings=wandb.Settings(start_method="thread"),
        )
        wandb.define_metric("flow/step")
        wandb.define_metric("permutation/step")
        wandb.define_metric("flow/*", step_metric="flow/step")
        wandb.define_metric("permutation/*", step_metric="permutation/step")
        dset = conf.data
        # save_synthetic_dataset(dset, conf, run_name)

        flow_dloader = torch.utils.data.DataLoader(
            dset, batch_size=conf.flow_batch_size, shuffle=True
        )
        perm_dloader = torch.utils.data.DataLoader(
            dset, batch_size=conf.permutation_batch_size, shuffle=True
        )

        trainer = conf.trainer(
            model=conf.model,
            dag=dset.dag,
            flow_dataloader=flow_dloader,
            perm_dataloader=perm_dloader,
        )
        trainer.train()


if __name__ == "__main__":
    main()
