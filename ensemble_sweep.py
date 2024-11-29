import hydra
import wandb
import os
import itertools
import torch
import random
from omegaconf import OmegaConf, DictConfig
from pprint import pprint
from random_word import RandomWords
from typing import List, Dict, Any


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


# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("get_torch_distribution", get_torch_distribution)
OmegaConf.register_new_resolver(
    "get_torch_distribution_args", get_torch_distribution_args
)
OmegaConf.register_new_resolver("get_permutations", get_permutations)


def init_run_dir(conf):
    # Handle preemption and resume
    run_name = str(conf.wandb.run_name)
    resume = True
    r = RandomWords()
    w1, w2 = r.get_random_word(), r.get_random_word()
    if run_name is None:
        run_name = f"{w1}_{w2}"
    else:
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


def setup_experiment_config(cfg: DictConfig, graph_size: int, 
                          structure: Dict, link_function: Dict, 
                          noise: Dict, variant: Dict, seed: int) -> DictConfig:
    """Configure a single experiment run"""
    # Deep copy the config
    cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))
    
    # Set up graph generator config
    cfg.data.graph_generator.num_nodes = graph_size
    cfg.data.graph_generator.graph_type = structure['type']
    cfg.data.graph_generator.seed = seed
    for k, v in structure['params'].items():
        setattr(cfg.data.graph_generator, k, v)

    # Set up data generation config
    if link_function['is_parametric']:
        cfg.data._target_ = "oslow.data.synthetic.parametric.AffineParametericDataset"
        cfg.data.link = link_function['type']
        # Keep the link_generator from default config
    else:
        cfg.data._target_ = "oslow.data.synthetic.nonparametric.AffineNonParametericDataset"
        # Set kernel parameters for nonparametric case
        for k, v in link_function['kernel_params'].items():
            setattr(cfg.data, k, v)
        # Remove link_generator and link if they exist since they're not used for nonparametric
        if hasattr(cfg.data, 'link_generator'):
            delattr(cfg.data, 'link_generator')
        if hasattr(cfg.data, 'link'):
            delattr(cfg.data, 'link')
    
    cfg.data.additive = variant['is_additive']
    
    # Set up noise generator
    cfg.data.noise_generator.noise_type = noise['type']
    for k, v in noise['params'].items():
        setattr(cfg.data.noise_generator, k, v)

    # Generate unique experiment name
    exp_name = f"{variant['name']}_{link_function['name']}_{structure['name']}_d{graph_size}_{noise['type']}_s{seed}"
    cfg.wandb.run_name = exp_name
    cfg.wandb.group = f"{variant['name']}_{link_function['name']}_{structure['name']}_d{graph_size}"
    
    return cfg


def run_single_experiment(cfg: DictConfig):
    """Run a single experiment with given configuration"""
    cfg = hydra.utils.instantiate(cfg)
    
    if cfg.test_run:
        pprint(OmegaConf.to_container(cfg, resolve=True))
        return
        
    cfg = init_run_dir(cfg)
    
    wandb.init(
        dir=cfg.out_dir,
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=cfg.wandb.run_name,
        group=cfg.wandb.group,
        id=cfg.wandb.run_id,
        resume="allow" if cfg.wandb.resume else False,
        settings=wandb.Settings(start_method="thread"),
    )
    
    # Set up wandb metrics
    wandb.define_metric("flow/step")
    wandb.define_metric("permutation/step")
    wandb.define_metric("flow/*", step_metric="flow/step")
    wandb.define_metric("permutation/*", step_metric="permutation/step")
    
    # Set up dataset and dataloaders
    dset = cfg.data
    flow_dloader = torch.utils.data.DataLoader(
        dset, batch_size=cfg.flow_batch_size, shuffle=True
    )
    flow_ensemble_dloader = torch.utils.data.DataLoader(
        dset, batch_size=cfg.flow_ensemble_batch_size, shuffle=True
    )
    
    model = cfg.model
    
    # Initialize and run trainer
    trainer = cfg.trainer(
        model=model,
        dag=dset.dag,
        flow_dataloader=flow_dloader,
        flow_ensemble_dataloader=flow_ensemble_dloader,
    )
    trainer.run()


@hydra.main(version_base=None, config_path="config", config_name="ensemble_sweep")
def main(cfg: DictConfig) -> None:
    """Main function to run all experiments"""
    random.seed(42)
    
    for graph_idx, graph_size in enumerate(cfg.experiment.graph_sizes, 1):
        num_sims = (cfg.experiment.simulations.large 
                   if graph_size >= 10 
                   else cfg.experiment.simulations.small)
        print(f"\nGraph Size d={graph_size} ({num_sims} simulations each)")
        # Generate all valid combinations based on the table
        valid_combinations = []
        for structure in cfg.experiment.structures:
            for link_function in cfg.experiment.link_functions:
                for noise in cfg.experiment.noise_types:
                    for variant in cfg.experiment.variants:
                        # Skip invalid combinations based on table constraints
                        if graph_size >= 10:
                            if structure['name'] != 'erdos_renyi' or variant['name'] != 'affine':
                                continue
                            if noise['name'] != 'normal':
                                continue
                        
                        if noise['name'] == 'laplace' and link_function['name'] != 'linear':
                            continue
                            
                        valid_combinations.append((structure, link_function, noise, variant))
        
        # Run experiments for each valid combination
        for combo_idx, (structure, link_function, noise, variant) in enumerate(valid_combinations, 1):
            print(f"\nCombination {combo_idx}/{len(valid_combinations)}:")
            print(f"  Structure: {structure['name']}")
            print(f"  Link Function: {link_function['name']}")
            print(f"  Noise: {noise['name']}")
            print(f"  Affine/Additive: {variant['name']}")
            
            for sim in range(num_sims):
                print(f"\n  Running simulation {sim + 1}/{num_sims}")
                seed = random.randint(0, 10000)
                exp_cfg = setup_experiment_config(
                    cfg, graph_size, structure, link_function, noise, variant, seed
                )
                
                try:
                    print(f"    Starting run: {exp_cfg.wandb.run_name}")
                    run_single_experiment(exp_cfg)
                    print(f"    Completed run: {exp_cfg.wandb.run_name}")
                except Exception as e:
                    print(f"    Error in run {exp_cfg.wandb.run_name}: {str(e)}")
                finally:
                    wandb.finish()


if __name__ == "__main__":
    main()