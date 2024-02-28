import hydra
import wandb
import os
import sys

from omegaconf import OmegaConf
from pprint import pprint
from random_word import RandomWords
import pandas as pd

sys.path.append("..")
_DIR = os.path.dirname(os.path.abspath(__file__))
_BASELINES_OUTPUT_DIR = os.path.join(_DIR, "results")

from oslow.data import OCDDataset
from baselines.source.base import AbstractBaseline


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


def save_results(conf, results_dict):
    result_path = os.path.join(_BASELINES_OUTPUT_DIR, "order_results.csv")
    dag_result_path = os.path.join(_BASELINES_OUTPUT_DIR, "dag_results.csv")
    file_path = dag_result_path if conf.DAG else result_path
    if not os.path.exists(file_path):
        pd.DataFrame([results_dict]).to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path)
        df.loc[len(df.index)] = results_dict
        df.to_csv(file_path, index=False)


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

    with open(config_yaml, "w") as fp:
        OmegaConf.save(config=conf, f=fp.name)

    return conf


@hydra.main(version_base=None, config_path="../config", config_name="baseline_conf")
def main(conf):
    log = OmegaConf.to_container(conf, resolve=True)["baseline"]
    conf = hydra.utils.instantiate(conf)

    if conf.test_run:
        pprint(OmegaConf.to_container(conf, resolve=True))
    else:
        conf = init_run_dir(conf)
        wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project,
            entity=conf.wandb.entity,
            config=OmegaConf.to_container(conf, resolve=True),
            name=conf.wandb.run_name,
            id=conf.wandb.run_id,
            resume="allow" if conf.wandb.resume else False,
            # compatible with hydra
            settings=wandb.Settings(start_method="thread"),
        )
        dset = conf.data
        assert isinstance(dset, OCDDataset), "Dataset must be an instance of OCDDataset"
        log = {**dset.log_dict(), **log}
        baseline: AbstractBaseline = conf.baseline
        baseline.set_dataset(dset)
        results = baseline.evaluate(conf.DAG)
        final_results = {**results, **log}
        save_results(conf, final_results)
        wandb.log(final_results)


if __name__ == "__main__":
    main()