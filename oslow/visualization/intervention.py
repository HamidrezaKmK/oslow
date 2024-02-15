import torch
import yaml
from lightning import seed_everything
import dypy as dy
from oslow.data import InterventionChainDataset
from oslow.training.callbacks.intervention import draw, draw_grid
import typing as th
from matplotlib import pyplot as plt


def visualize_results(
    data_config_path: str,
    config_path: th.Optional[str] = None,
    checkpoint_path: th.Optional[str] = None,
    num_interventions: int = 1000,
    num_samples: int = 20,
    k: float = 8.0,
    percentile: float = 0.99,
    limit_y: float = 0,
    limit_ys: th.Optional[th.Tuple[float, float]] = None,
    target: th.Optional[th.Union[th.List[int], int]] = None,
):
    # load model and seeds
    if config_path is not None:
        config = yaml.safe_load(open(config_path, "r"))
        seed_everything(config["seed_everything"])
        model = dy.eval(config["model"]["class_path"])(**config["model"]["init_args"])
        if checkpoint_path is not None:
            model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu")["state_dict"]
            )
        flow = model.model.flow

    data_config = yaml.safe_load(open(data_config_path, "r"))
    dataset = InterventionChainDataset(**data_config["init_args"]["dataset_args"])
    n = dataset.data.shape[-1]
    values = torch.linspace(-k, k, num_interventions)
    pred_means, pred_stds = None, None
    if config_path is not None:
        with torch.no_grad():
            pred_samples = flow.do(0, values, num_samples=num_samples)
            pred_means = pred_samples.mean(-2)
            pred_stds = pred_samples.std(-2)
    gt_samples = dataset.do(0, values, num_samples=num_samples)
    gt_mean = gt_samples.mean(-2)
    gt_std = gt_samples.std(-2)

    if percentile > 0:
        cis = [(1 - percentile) / 2, 1 - (1 - percentile) / 2]
        icis = dataset.base_distribution.icdf(torch.tensor(cis)).detach().cpu()
    if target is None or not isinstance(target, int):
        _ = draw_grid(
            k=k,
            n=n,
            values=values,
            limit_ys=limit_ys,
            pred_means=pred_means,
            pred_stds=pred_stds,
            gt_means=gt_mean,
            gt_stds=gt_std,
            target=target,
            limit_y=limit_y,
            percentile=percentile,
            icis=icis,
        )
    else:
        _ = draw(
            fig=plt.figure(figsize=(8, 8)),
            k=k,
            n=n,
            values=values,
            limit_ys=limit_ys,
            pred_means=pred_means,
            pred_stds=pred_stds,
            gt_means=gt_mean,
            gt_stds=gt_std,
            target=target,
            percentile=percentile,
            icis=icis,
            limit_y=limit_y,
        )
