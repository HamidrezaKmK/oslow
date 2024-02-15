import hydra
import wandb
import os
import yaml

from omegaconf import OmegaConf
from pprint import pprint
import oslow.config as config_ref
from oslow.config import MainConfig, DataConfig, ModelConfig, TrainingConfig
from random_word import RandomWords
from oslow.data.synthetic.graph_generator import GraphGenerator
from oslow.data.synthetic.parametric import AffineParametericDataset
from oslow.data.synthetic.nonparametric import AffineNonParametericDataset
from oslow.data.real_world.sachs import SachsOCDDataset
from oslow.data.real_world.syntren import SyntrenOCDDataset
from typing import Union
from oslow.data.base_dataset import OCDDataset
from oslow.models.oslow import OSlow
from oslow.training.trainer import Trainer
import torch

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_torch_distribution(distr_name):
    if distr_name == 'laplace':
        return 'torch.distributions.Laplace'
    elif distr_name == 'normal':
        return 'torch.distributions.Normal'
    elif distr_name == 'uniform':
        return 'torch.distributions.Uniform'
    else:
        ValueError(f"Distribution {distr_name} cannot be resolved!")

def get_torch_distribution_args(distr_name):
    if distr_name == 'laplace':
        return [0., 1.]
    elif distr_name == 'normal':
        return [0., 1.]
    elif distr_name == 'uniform':
        return [0., 1.]
    else:
        ValueError(f"Distribution {distr_name} cannot be resolved!")
    


# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("get_torch_distribution", get_torch_distribution)
OmegaConf.register_new_resolver("get_torch_distribution_args", get_torch_distribution_args)

def init_run_dir(conf: MainConfig) -> MainConfig:
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
            old_conf = MainConfig(**yaml.load(fp, Loader=yaml.Loader))
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
    with open(config_yaml, "w") as fp:
        yaml.dump(conf.model_dump(), fp, default_flow_style=False)

    return conf


def instantiate_data(conf: Union[DataConfig, OCDDataset]):
    """
    Instantiate the dataset according to the data configuration specified
    and then create an appropriate training dataloader and return it.
    """

    synth_cond1 = isinstance(
        conf.dataset, config_ref.ParametricSyntheticConfig)
    synth_cond2 = isinstance(
        conf.dataset, config_ref.NonParametricSyntheticConfig)
    if isinstance(conf, OCDDataset):
        dset = conf
    elif synth_cond1 or synth_cond2:
        graph_generator = GraphGenerator(
            num_nodes=conf.dataset.graph.num_nodes,
            seed=conf.dataset.graph.seed,
            graph_type=conf.dataset.graph.graph_type,
            enforce_ordering=conf.dataset.graph.enforce_ordering,
        )
        graph = graph_generator.generate_dag()

        if synth_cond1:
            dset = AffineParametericDataset(
                num_samples=conf.dataset.num_samples,
                graph=graph,
                noise_generator=conf.dataset.noise_generator,
                link_generator=conf.dataset.link_generator,
                link=conf.dataset.link,
                perform_normalization=conf.dataset.perform_normalization,
                additive=conf.dataset.additive,
                post_non_linear_transform=conf.dataset.post_non_linear_transform,
                standard=conf.standard,
            )
        else:
            dset = AffineNonParametericDataset(
                num_samples=conf.dataset.num_samples,
                graph=graph,
                noise_generator=conf.dataset.noise_generator,
                perform_normalization=conf.dataset.perform_normalization,
                additive=conf.dataset.additive,
                post_non_linear_transform=conf.dataset.post_non_linear_transform,
                standard=conf.standard,
            )
    elif isinstance(conf.dataset, config_ref.RealworldConfig):
        if conf.dataset.name == "sachs":
            dset = SachsOCDDataset(
                standard=conf.standard,
                name=conf.dataset.name,
            )
        else:
            raise ValueError(f"Unknown real world dataset {conf.dataset.name}")
    elif isinstance(conf.dataset, config_ref.SemiSyntheticConfig):
        if conf.dataset.name == "syntren":
            dset = SyntrenOCDDataset(
                standard=conf.standard,
                data_id=conf.dataset.data_id,
            )
        else:
            raise ValueError(
                f"Unknown semi synthetic dataset {conf.dataset.name}")
    else:
        raise ValueError(f"Unknown dataset type {conf.dataset}")

    return dset


def instantiate_model(conf: ModelConfig):
    return OSlow(
        in_features=conf.in_features,
        layers=conf.layers,
        dropout=conf.dropout,
        residual=conf.residual,
        activation=conf.activation,
        additive=conf.additive,
        num_transforms=conf.num_transforms,
        normalization=conf.normalization,
        base_distribution=conf.base_distribution,
        ordering=conf.ordering,
        num_post_nonlinear_transforms=conf.num_post_nonlinear_transforms,
    )


def instantiate_trainer(conf: TrainingConfig, model, flow_dloader, perm_dloader, dag):
    return Trainer(
        model=model,
        dag=dag,
        flow_dataloader=flow_dloader,
        perm_dataloader=perm_dloader,
        flow_optimizer=conf.flow_optimizer,
        permutation_optimizer=conf.permutation_optimizer,
        flow_frequency=conf.scheduler.flow_frequency,
        permutation_frequency=conf.scheduler.permutation_frequency,
        flow_lr_scheduler=conf.scheduler.flow_lr_scheduler,
        permutation_lr_scheduler=conf.scheduler.permutation_lr_scheduler,
        device=conf.device,
        temperature=conf.temperature,
        temperature_scheduler=conf.temperature_scheduler,
        max_epochs=conf.max_epochs,
        permutation_learning_config=conf.permutation,
        birkhoff_config=conf.brikhoff,
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: MainConfig):
    conf = hydra.utils.instantiate(conf)
    conf = MainConfig(**OmegaConf.to_container(conf))

    if conf.test_run:
        pprint(conf.model_dump())
    else:
        conf = init_run_dir(conf)
        wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project,
            entity=conf.wandb.entity,
            config=conf.model_dump(),
            name=conf.wandb.run_name,
            id=conf.wandb.run_id,
            resume="allow" if conf.wandb.resume else False,
            # compatible with hydra
            settings=wandb.Settings(start_method="thread"),
        )
        logging.info("Starting ...")
        wandb.define_metric("flow/step")
        wandb.define_metric("permutation/step")
        wandb.define_metric("flow/*", step_metric="flow/step")
        wandb.define_metric("permutation/*", step_metric="permutation/step")
        logging.info("Instantiate dataset ...")
        dset = instantiate_data(conf.data)
        logging.info("Instantiate data loaders ...")
        flow_dloader = torch.utils.data.DataLoader(
            dset,
            batch_size=conf.trainer.flow_batch_size,
            shuffle=True,
        )
        perm_dloader = torch.utils.data.DataLoader(
            dset,
            batch_size=conf.trainer.permutation_batch_size,
            shuffle=True,
        )
        logging.info("Instantiate model ...")
        model = instantiate_model(conf.model)

        logging.info("Instantiate trainer...")
        trainer = instantiate_trainer(
            conf.trainer, model, flow_dloader, perm_dloader, dag=dset.dag
        )
        logging.info("Start training ...")
        trainer.train()


if __name__ == "__main__":
    main()
