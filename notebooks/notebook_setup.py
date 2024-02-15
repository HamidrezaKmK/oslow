"""
Script meant to be imported at the beginning of experimental notebooks.
Sets random seeds and moves the notebook's working directory to the project root.
"""
import os
import random
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from oslow.models.normalization import ActNorm
from oslow.models.oslow import OSlow
from torch.utils.data import TensorDataset, DataLoader

# Configure file system
notebook_path = Path(__file__).parent
project_root = notebook_path.parent

# Configure PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(0)

# Set matplotlib colour theme
plt.style.use('seaborn-v0_8-pastel')


# Function to smooth the loss graphs
def smooth_graph(histories, window_size=100):
    smoothed = {}
    for key, history in histories.items():
        smoothed[key] = []
        for i in range(window_size, len(history)):
            smoothed[key].append(sum(history[max(0, i - window_size) : i + 1])/window_size)
    return smoothed

def create_new_set_of_models(
    additive = False,
    num_transforms = 1,
    normalization = ActNorm,
    base_distribution = torch.distributions.Normal(loc=0, scale=1),
    use_standard_ordering = False,
    num_post_nonlinear_transforms = 0,
    single_ordering: Optional[str] = None,
    **post_non_linear_transform_kwargs,
):
    if single_ordering is not None:
        all_models = {
            single_ordering: None,
        }
    else:
        all_models = {
            '012': None,
            '021': None,
            '102': None,
            '120': None,
            '201': None,
            '210': None,
        }

    for ordering in all_models.keys():
        order = [int(x) for x in ordering]
        all_models[ordering] = OSlow(
            in_features=3,
            layers=[100, 100],
            dropout=None,
            residual=False,
            activation=torch.nn.LeakyReLU(),
            additive=additive,
            num_transforms=num_transforms,
            normalization=normalization,
            base_distribution=base_distribution,
            ordering=None if use_standard_ordering else torch.IntTensor(order),
            num_post_nonlinear_transforms=num_post_nonlinear_transforms,
            **post_non_linear_transform_kwargs,
        )
    
    return all_models if single_ordering is None else all_models[single_ordering]
    

def train_models_and_get_histories(
    all_models, 
    dset, 
    batch_size=128, 
    lr=0.005, 
    epoch_count=100,
    use_standard_ordering = False,
):
        
    tensor_samples = torch.tensor(dset.samples.values).float()
    torch_dataset = TensorDataset(tensor_samples)
    torch_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)

    all_histories_laplace = {}
    for key, model in all_models.items():
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        all_histories_laplace[key] = []
        progress_bar = tqdm(range(epoch_count), desc="training model {}".format(key))

        for epoch in progress_bar:
            for batch, in torch_dataloader:
                batch = batch.to(device)
                if not use_standard_ordering:
                    loss = -model.log_prob(batch).mean()
                else:
                    order = [int(x) for x in key]
                    # create a permutation matrix from the order
                    permutation_matrix = torch.zeros((3, 3))
                    for i, j in enumerate(order):
                        permutation_matrix[i, j] = 1
                    loss = -model.log_prob(batch, perm_mat=permutation_matrix).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                all_histories_laplace[key].append(loss.item())
    
    return all_histories_laplace

def update_dict(
    base_dict: dict,
    **kwargs,
):
    new_dict = base_dict.copy()
    for key, value in kwargs.items():
        new_dict[key] = value
    return new_dict