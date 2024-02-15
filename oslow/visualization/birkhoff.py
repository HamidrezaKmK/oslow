from typing import Optional, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import torch
from oslow.models.oslow import OSlow
from functools import lru_cache
from sklearn.decomposition import PCA
from oslow.training.permutation import PermutationLearningModule, PermutationMatrixLearningModuleWithBuffer
import networkx as nx
from oslow.evaluation import backward_relative_penalty


@lru_cache(maxsize=128)
def get_all_permutation_matrices(num_nodes: int) -> torch.Tensor:
    # return a tensor of shape (num_nodes!, num_nodes, num_nodes) where the first dimension is all the permutation matrices

    return torch.tensor(
        np.array(
            [
                np.eye(num_nodes)[list(p)]
                for p in np.ndindex(*([num_nodes] * num_nodes))
                if np.all(np.sum(np.eye(num_nodes)[list(p)], axis=0) == 1)
                and np.all(np.sum(np.eye(num_nodes)[list(p)], axis=1) == 1)
            ]
        )
    )


def get_label_from_permutation_matrix(permutation_matrix: torch.Tensor) -> str:
    return "".join([str(int(i)) for i in permutation_matrix.argmax(dim=-1)])


def visualize_birkhoff_polytope(
    permutation_model: PermutationLearningModule,
    num_samples: int,
    data: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    flow_model: Optional[OSlow] = None,
    device: str = "cpu",
    print_legend: bool = True,
    dag: Optional[nx.DiGraph] = None,
    temperature: float = 1.0,
):
    """
    This function is intended for either debugging purposes or just for visualizing the training process of the model itself.

    Given a permutation_model and number of samples, this function will sample from the model and visualize the sampled permutations in the Birkhoff polytope with appropriate coloring.
    The average samples that are generated will also be visualized with a red cross indicating the state of the model. As the training progresses, we expect the cross to gradually collapse on a correct ordering.

    If data and flow_model are provided, the log_prob of the data will be calculated for each permutation and the value will be written on the plot.
    For every permutation, a text would appear in format {average_log_prob} : #{number_of_permutations} where average_log_prob is the average log_prob of the data
    if it were evaluated at that permutation and and number_of_permutations is the number of samples that are closest to the permutation.

    Args:
        permutation_model: the model to sample from
        num_samples: number of samples to generate
        data: the data to evaluate the log_prob of the model, this is optional.
        flow_model: the flow model to evaluate the log_prob of the data, this is optional.
        device: the device to run the model and the flow model on
    Returns:
        a numpy array that can be converted to an image representing the visualization of the Birkhoff polytope
    """
    permutation_model.to(device)
    sampled_permutations = permutation_model.sample_hard_permutations(
        num_samples, gumbel_std=temperature)

    # (1) Handle the backbone and the PCA transform of the Birkhoff polytope
    d = sampled_permutations.shape[-1]
    all_permutation_matrices = get_all_permutation_matrices(d).to(device)
    # concatenate all_permutation_matrices with backbone and train a PCA
    backbone = all_permutation_matrices
    pca = PCA(n_components=2, random_state=42)
    pca.fit(backbone.reshape(backbone.shape[0], -1).cpu())

    # (2) find the closest reference permutation matrix for each sampled permutation
    flattened_samples = sampled_permutations.reshape(
        sampled_permutations.shape[0], -1)
    flattened_references = all_permutation_matrices.reshape(
        all_permutation_matrices.shape[0], -1
    )
    # for every flattened samples find the closest reference
    closest_references = torch.argmin(
        torch.norm(
            flattened_samples.unsqueeze(1) - flattened_references.unsqueeze(0), dim=-1
        ),
        dim=-1,
    )
    img_data = None

    try:
        # plot the log_prob values of the closest references using the data if available
        fig, ax = plt.subplots()

        # plot backbone without showing it in the legend
        ax.scatter(
            pca.transform(backbone.reshape(backbone.shape[0], -1).cpu())[:, 0],
            pca.transform(backbone.reshape(backbone.shape[0], -1).cpu())[:, 1],
            alpha=0,
        )

        if isinstance(data, torch.Tensor):
            data = [data]

        for i, permutation in enumerate(all_permutation_matrices):
            txt = "NA"
            lbl = get_label_from_permutation_matrix(permutation)

            if isinstance(permutation_model, PermutationMatrixLearningModuleWithBuffer):
                idx = permutation_model._get_in_buffer_index(
                    permutation.unsqueeze(0))[0].item()
                if idx != -1:
                    txt = f"*{permutation_model.permutation_buffer_scores[idx].item():.2f}"
                # check if there is any matrix in my_buffer that is equal to permutation
            average_log_prob = 0.0
            if (closest_references == i).any():
                close_samples = sampled_permutations[closest_references == i]
                alpha_value = max(
                    0.3, 1.0 / (closest_references == i).sum().item())
                ax.scatter(
                    pca.transform(
                        close_samples.reshape(
                            close_samples.shape[0], -1).cpu()
                    )[:, 0],
                    pca.transform(
                        close_samples.reshape(
                            close_samples.shape[0], -1).cpu()
                    )[:, 1],
                    label=lbl,
                    alpha=alpha_value,
                )

                if data is not None and flow_model is not None:
                    flow_model.to(device)
                    with torch.no_grad():
                        log_prob_sum = 0
                        num_data = 0
                        for data_batch in data:
                            data_batch = data_batch.to(device)
                            log_prob_sum += flow_model.log_prob(
                                data_batch.float(), perm_mat=permutation.float()
                            ).sum()
                            num_data += data_batch.shape[0]
                        average_log_prob = log_prob_sum / num_data
                num_samples = (closest_references == i).sum()
                if num_samples > 0:
                    if txt == "NA":
                        txt = f"{average_log_prob:.2f}:#{num_samples}"
                    else:
                        txt = f"{txt}:{average_log_prob:.2f}:#{num_samples}"

            if dag is not None:
                perm_list = torch.argmax(
                    permutation, dim=-1).cpu().numpy().tolist()
                txt = f"{txt}:{backward_relative_penalty(perm_list, dag):.2f}"

            x, y = (
                pca.transform(permutation.reshape(1, -1).cpu())[0, 0],
                pca.transform(permutation.reshape(1, -1).cpu())[0, 1],
            )
            # count the number of entries in the closest references that is equal to i
            # write a text on (x, y) with value of average_log_prob
            ax.text(x, y, txt, fontsize=6)

        mean_samples = torch.mean(sampled_permutations, dim=0)
        ax.scatter(
            pca.transform(mean_samples.reshape(1, -1).cpu())[:, 0],
            pca.transform(mean_samples.reshape(1, -1).cpu())[:, 1],
            label="mean",
            marker="x",
            color="red",
        )
        if print_legend:
            ax.legend()

        # draw everything to the figure for conversion
        fig.canvas.draw()
        # convert the figure to a numpy array
        img_data = np.fromstring(
            fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    finally:
        plt.close()

    return img_data
