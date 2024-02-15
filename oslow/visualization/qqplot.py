import torch
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np


def qqplot(
    model_samples: torch.Tensor,
    data_samples: torch.Tensor,
    reject_outliers_factor: float = 3.0,
    quantile_count: int = 100,
    image_size: Tuple = (5, 5),
):
    """
    Args:
        model_samples: samples from the model distribution (z's) of shape (batch_size, D)
        data_samples: samples from the data distribution (x's) of shape (batch_size, D)
        reject_outliers_factor: the factor to multiply the standard deviation with to reject outliers
        quantile_count: the number of quantiles to plot
        image_size: the size of the image to be returned
    Returns:
        a list of numpy arrays that can be converted to images representing the qqplot corresponding to each column of model_samples and data_samples
        the i'th element of the list corresponds to the qqplot of the i'th column.
    """
    if model_samples.shape[1] != data_samples.shape[1]:
        raise ValueError(
            "model_samples and data_samples must have the same number of columns"
        )

    res = []

    for i in range(model_samples.shape[1]):
        x_samples = model_samples[:, i].detach().cpu().numpy().flatten()
        y_samples = data_samples[:, i].detach().cpu().numpy().flatten()

        # (1) filter out all the nan or inf values and everything considered as outliers in x_samples
        potential_nan = np.isnan(x_samples) | np.isinf(x_samples)
        x_samples = x_samples[~potential_nan]
        potential_outliers = np.abs(
            x_samples - np.mean(x_samples)
        ) > reject_outliers_factor * np.std(x_samples)
        x_samples = x_samples[~potential_outliers]

        # (2) filter out all the nan or inf values and everything considered as outliers in y_samples
        potential_nan = np.isnan(y_samples) | np.isinf(y_samples)
        y_samples = y_samples[~potential_nan]
        potential_outliers = np.abs(
            y_samples - np.mean(y_samples)
        ) > reject_outliers_factor * np.std(y_samples)
        y_samples = y_samples[~potential_outliers]

        fig, ax = plt.subplots()
        # customize the image size if needed
        if image_size:
            fig.set_size_inches(image_size[0], image_size[1])
        try:
            ax.set_title(f"{i+1}'th column")
            ax.set_xlabel("model_samples")
            ax.set_ylabel("data_samples")
            mn = min(np.min(x_samples), np.min(y_samples))
            mx = max(np.max(x_samples), np.max(y_samples))
            ax.plot(
                np.linspace(mn, mx, 100),
                np.linspace(mn, mx, 100),
                c="red",
                alpha=0.2,
                label="y=x",
            )
            ax.text(
                0,
                0,
                f"Outliers: {np.sum(potential_nan)}/{len(potential_nan)}",
                fontsize=10,
                color="red",
            )
            x_samples = np.sort(x_samples)
            y_samples = np.sort(y_samples)

            # get equally seperated samples according to the quantile_count from the sorted x_samples
            x_quantiles = x_samples[
                np.linspace(0, len(x_samples) - 1, quantile_count).astype(int)
            ]
            y_quantiles = y_samples[
                np.linspace(0, len(y_samples) - 1, quantile_count).astype(int)
            ]

            ax.scatter(x_quantiles, y_quantiles, s=1, alpha=1.0, label="quantiles")

            ax.legend()

            # draw everything to the figure for conversion
            fig.canvas.draw()
            # convert the figure to a numpy array
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        finally:
            plt.close()

        res.append(data)

    return res
