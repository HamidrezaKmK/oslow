import sys
import os
sys.path.append('../')
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from notebooks.notebook_setup import device, smooth_graph, create_new_set_of_models, train_models_and_get_histories, update_dict
from oslow.models.oslow import OSlowTest, OSlow
from oslow.data.synthetic.graph_generator import GraphGenerator
from oslow.data.synthetic.utils import RandomGenerator
from oslow.data.synthetic.parametric import AffineParametericDataset
from oslow.models.normalization import ActNorm
from oslow.training.trainer import Trainer
from oslow.training.permutation_learning.buffered_methods import GumbelTopK
from oslow.visualization.birkhoff import get_all_permutation_matrices
from oslow.evaluation import backward_relative_penalty
from oslow.training.permutation_learning.initialization import uniform_gamma_init_func
import wandb
import numpy as np
import matplotlib as mpl


################################################
################################################
################################################
# NOTE: set these.
ground_truth_ordering = [2, 0, 3, 1]
ground_truth_ordering_str = '2031'
# Possible options: "sinusoid", "cubic", "linear", "square", "absolute", "sigmoid"
link_function = "linear"

# Possible options: "exp", "softplus", "x_plus_sin", "nonparametric", "sigmoid", "tanh", "spline"
pnl_transform = "softplus" 

# Possible options: "normal", "laplace", "uniform"
exogenous_noise = 'normal'
################################################
################################################
################################################


# Create output directory
os.makedirs("ensemble_results", exist_ok=True)

# Setup device and seeds
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.random.manual_seed(101)
print(f"Using device: {device}")

# Hyperparameters
num_samples = 1000
permutation_batch_size = 128
flow_batch_size = 128
epochs = 500
flow_lr = 0.001
perm_lr = 0.00001
flow_freq = 1
perm_freq = 0
num_nodes = 4

# Generate graph
graph_generator = GraphGenerator(
    num_nodes=num_nodes,
    seed=12,
    graph_type="full",
    enforce_ordering=ground_truth_ordering,
)

# Setup generators
# play with the exogeneous noise, maybe use 'laplace' or 'uniform' instead of Gaussian noise
gaussian_noise_generator = RandomGenerator(exogenous_noise, seed=30, loc=0, scale=1)
link_generator = RandomGenerator('uniform', seed=1100, low=1, high=1)

generated_dset = AffineParametericDataset(
    num_samples=num_samples,
    graph_generator=graph_generator,
    noise_generator=gaussian_noise_generator,
    link_generator=link_generator,
    link=link_function,  
    perform_normalization=True, # should always be True
    standard=False,  # don't use this for PNL, should normalize as we go
    additive=True,  # to make ANM
    post_non_linear_transform=pnl_transform,  # PNL
)

# Dataset wrapper
class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return len(self.tensor)

# Create dataloaders
dataset = CustomTensorDataset(torch.tensor(generated_dset.samples.values).float())
flow_dataloader = DataLoader(dataset, batch_size=flow_batch_size, shuffle=True)
permutation_dataloader = DataLoader(dataset, batch_size=permutation_batch_size, shuffle=True)

# Initialize model
model = OSlow(in_features=num_nodes,
              layers=[128, 64, 128],
              dropout=None,
              residual=False,
              activation=torch.nn.LeakyReLU(),
              additive=True,   # True for ANM
              num_transforms=1,
              num_post_nonlinear_transforms=3,
              normalization=ActNorm,
              base_distribution=torch.distributions.Normal(loc=0, scale=1),
              ordering=None)

# Setup optimizers
def flow_optimizer(params): return torch.optim.AdamW(params, lr=flow_lr)
def perm_optimizer(params): return torch.optim.AdamW(params, lr=perm_lr)

# Training configuration
temperature_scheduler = 'constant'
temperature = 1.0

perm_module = lambda in_features: GumbelTopK(
    in_features, 
    num_samples=num_samples, 
    buffer_size=10, 
    buffer_update=10, 
    initialization_function=uniform_gamma_init_func
)

# Initialize trainer
trainer = Trainer(
    model=model,
    dag=generated_dset.dag,
    flow_dataloader=flow_dataloader,
    perm_dataloader=permutation_dataloader,
    flow_optimizer=flow_optimizer,
    permutation_optimizer=perm_optimizer,
    flow_frequency=flow_freq,
    temperature=temperature,
    temperature_scheduler=temperature_scheduler,
    permutation_frequency=perm_freq,
    max_epochs=epochs,
    flow_lr_scheduler=torch.optim.lr_scheduler.ConstantLR,
    permutation_lr_scheduler=torch.optim.lr_scheduler.ConstantLR,
    permutation_learning_module=perm_module,
    device=device
)

# Training
wandb.init(
    project="order_discovery",
    name='oslow-ensemble',
    tags=[
        f"num_nodes-{num_nodes}",
        f"epochs-{epochs}",
        f"base-temperature-{temperature}",
        f"temperature-scheduling-{temperature_scheduler}",
        link_function,
    ],
)
trainer.train()
wandb.finish()

# Evaluation
model.eval()
all_perms = get_all_permutation_matrices(4)

all_scores = []
all_backward = []

# Save permutation results to file
with open(f"ensemble_results/PNL-ANM_results_{link_function}_{ground_truth_ordering_str}_trueordering.txt", "w") as f:
    f.write("Permutation Results:\n\n")
    with torch.no_grad():
        for perm in all_perms:
            perm = perm.float().to(device)
            all_log_probs = []
            for x in flow_dataloader:
                x = x.to(device)
                perm_mat = perm.unsqueeze(0).repeat(x.shape[0], 1, 1)
                all_log_probs.append(
                    model.log_prob(x, perm_mat=perm_mat)
                    .mean()
                    .item()
                )
            # TODO: use dim=0.
            perm_list = [x for x in torch.argmax(perm, dim=0).cpu().numpy().tolist()]
            perm_list_formatted = "".join([str(x) for x in perm_list])
            score = sum(all_log_probs) / len(all_log_probs)
            backward = backward_relative_penalty(perm_list, generated_dset.dag)
            all_scores.append(-score)
            all_backward.append(backward)
            
            # Write results to file
            print(f"Ground Truth: {ground_truth_ordering_str}\n")
            print(f"Permutation: {perm_list_formatted}\n")
            print(f"Log Prob: {score}\n")
            print(f"Backward count: {backward}\n")
            print("-----\n")
            
            f.write(f"Permutation: {perm_list_formatted}\n")
            f.write(f"Log Prob: {score}\n")
            f.write(f"Backward count: {backward}\n")
            f.write("-----\n")

LIGHT_COLORS = {
    "blue": (0.237808, 0.688745, 1.0),
    "red": (1.0, 0.519599, 0.309677),
    "green": (0.0, 0.790412, 0.705117),
    "pink": (0.936386, 0.506537, 0.981107),
    "yellow": (0.686959, 0.690574, 0.0577502),
    "black": "#535154",
}

cm = 1 / 2.54
FIG_WIDTH = 17 * cm
FONT_SIZE = 10
LEGEND_SIZE = 8

plt.figure(figsize=(8, 6))
plt.scatter(all_backward, all_scores, s=50, marker="o", alpha=0.6, color='blue')
plt.xlabel("CBC (Causal Backward Count)")
plt.ylabel("Negative Log-likelihood")
plt.title(f"Permutation Performance: {link_function} function")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"ensemble_results/PNL-ANM_ensemble_{link_function}_{ground_truth_ordering_str}ordering.png", dpi=300, bbox_inches='tight')
plt.close()

# Also save the raw data for the plot
with open("ensemble_results/plot_data.txt", "w") as f:
    f.write("CBC,NegativeLogLikelihood\n")
    for b, s in zip(all_backward, all_scores):
        f.write(f"{b},{s}\n")