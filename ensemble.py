import hydra
import wandb
import os
import itertools
import torch

from omegaconf import OmegaConf
from pprint import pprint
from random_word import RandomWords
from typing import List


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


@hydra.main(version_base=None, config_path="config", config_name="ensemble")
def main(conf):
    conf = hydra.utils.instantiate(conf)
    if conf.test_run:
        pprint(OmegaConf.to_container(conf, resolve=True))
    else:
        conf = init_run_dir(conf)
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
        flow_dloader = torch.utils.data.DataLoader(
            dset, batch_size=conf.flow_batch_size, shuffle=True
        )
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


if __name__ == "__main__":
    main()
