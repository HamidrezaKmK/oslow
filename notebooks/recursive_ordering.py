import sys
sys.path.append('../')
import torch
from torch.utils.data import TensorDataset, DataLoader
import random
import logging
from tqdm import tqdm
from oslow.models.oslow import OSlow
from oslow.data.synthetic.graph_generator import GraphGenerator
from oslow.data.synthetic.utils import RandomGenerator
from oslow.data.synthetic.parametric import AffineParametericDataset
from oslow.data.synthetic.nonparametric import AffineNonParametericDataset
from oslow.models.normalization import ActNorm
from itertools import permutations

# Set up logging
logging.basicConfig(filename='causal_ordering_test_results120.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    logging.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    logging.info("Using CPU")
    raise SystemExit("GPU not available")

# Function definitions
def sample_permutations(remaining, num_samples=10):
    if len(remaining) == 1:
        return [[remaining[0]]] * num_samples
    sampled_permutations = []
    for _ in range(num_samples):
        perm = list(remaining)
        random.shuffle(perm)
        sampled_permutations.append(perm)
    return sampled_permutations

def create_oslow_model_with_ordering(ordering, additive=False, base_distribution=None):
    if base_distribution is None:
        base_distribution = torch.distributions.Normal(loc=0, scale=1)
    
    return OSlow(
        in_features=len(ordering),
        layers=[100, 100],
        dropout=None,
        residual=False,
        activation=torch.nn.LeakyReLU(),
        additive=additive,
        num_transforms=1,
        normalization=ActNorm,
        base_distribution=base_distribution,
        ordering=torch.tensor(ordering) # this should be set in log_prob perm mat
    )

def initialize_models_and_optimizers(determined_ordering, remaining_covariates, num_total_covariates, model_params):
    all_permutations = []
    for i in range(len(remaining_covariates)):
        start_covariate = remaining_covariates[i]
        if i == 0:
            permutation_remaining_covariates = remaining_covariates[1:]
        elif i == len(remaining_covariates) - 1:
            permutation_remaining_covariates = remaining_covariates[:i]
        else:
            permutation_remaining_covariates = remaining_covariates[:i] + remaining_covariates[i+1:]
        
        samples = sample_permutations(permutation_remaining_covariates, num_samples=10)
        all_permutations.extend([[start_covariate] + sample for sample in samples])
    
    models = {}
    histories = {}

    for perm in all_permutations:
        full_perm = determined_ordering + perm
        perm_key = tuple(full_perm)

        model = create_oslow_model_with_ordering(full_perm, **model_params).to(device)

        if perm_key not in models:
            models[perm_key] = []
            histories[perm_key] = []

        models[perm_key].append(model)
        histories[perm_key].append([])

        perm_matrix = torch.zeros((num_total_covariates, num_total_covariates))
        for i, j in enumerate(full_perm):
            perm_matrix[i, j] = 1
        perm_matrix = perm_matrix.to(device)

        models[perm_key][-1].perm_matrix = perm_matrix

    return models, histories

def determine_ordering(remaining_covariates, dataloader, model_params, training_params, determined_ordering=None, num_total_covariates=None, true_ordering=None, depth=0):
    if determined_ordering is None:
        determined_ordering = []
    if num_total_covariates is None:
        num_total_covariates = len(remaining_covariates)

    if len(remaining_covariates) == 1:
        return determined_ordering + remaining_covariates

    logging.info(f"\nCurrent stage of recursion at depth {depth}:")
    logging.info(f"  Ground truth ordering: {true_ordering}")
    logging.info(f"  Fixed ordering so far: {determined_ordering}")
    logging.info(f"  Remaining covariates: {remaining_covariates}")

    models, histories = initialize_models_and_optimizers(
        determined_ordering, remaining_covariates, num_total_covariates, model_params
    )

    for perm_key in tqdm(models, desc=f"Training models for {len(remaining_covariates)} remaining covariates"):
        for model_index in range(len(models[perm_key])):
            model = models[perm_key][model_index]
            optimizer = torch.optim.AdamW(model.parameters(), lr=training_params['lr'], weight_decay=0.5)
            
            for epoch in range(training_params['epoch_count']):
                for batch, in dataloader:
                    # sample the remaining covariates in the order specified by the permutation
                    
                    batch = batch.to(device)
                    log_prob = model.log_prob(batch, perm_mat=[0,2,1]).mean()
                    loss = -log_prob
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    histories[perm_key][model_index].append(log_prob.item())
    
    starting_covariate_avg_log_probs = {}
    for covariate in remaining_covariates:
        relevant_perms = [perm for perm in models.keys() if perm[len(determined_ordering)] == covariate]
        total_log_prob = 0
        total_models = 0
        
        for perm in relevant_perms:
            for model_history in histories[perm]:
                total_log_prob += model_history[-1]
                total_models += 1
        
        if total_models > 0:
            avg_log_prob = total_log_prob / total_models
        else:
            avg_log_prob = float('-inf')
        
        starting_covariate_avg_log_probs[covariate] = avg_log_prob

    next_covariate = max(starting_covariate_avg_log_probs, key=starting_covariate_avg_log_probs.get)
    
    logging.info(f"\nSanity Check Results at depth {depth}:")
    logging.info(f"  Average log probabilities for each starting covariate:")
    for covariate, avg_log_prob in starting_covariate_avg_log_probs.items():
        logging.info(f"    Covariate {covariate}: {avg_log_prob}")
    logging.info(f"  Best covariate at this stage: {next_covariate}")
    logging.info(f"  Current ordering so far (including new covariate): {determined_ordering + [next_covariate]}")
    logging.info(f"  True ordering so far (including new covariate): {true_ordering[:depth+1]}")
    
    remaining_covariates = [i for i in remaining_covariates if i != next_covariate]
    return determine_ordering(remaining_covariates, dataloader, model_params, training_params, determined_ordering + [next_covariate], num_total_covariates, true_ordering, depth+1)

def run_single_test(dataset_name, dataset, true_ordering, num_covariates):
    logging.info(f"\nTesting on {dataset_name} dataset:")
    
    if dataset_name == "laplace_linear":
        model_params = {
            "additive": True,
            "base_distribution": torch.distributions.Laplace(loc=0, scale=1)
        }
    elif "nonparametric_additive" in dataset_name:
        model_params = {"additive": True}
    else:
        model_params = {}
    
    training_params = {
        "batch_size": 512,
        "lr": 0.005,
        "epoch_count": 30
    }

    tensor_samples = torch.tensor(dataset.samples.values).float().clone().detach()
    torch_dataset = TensorDataset(tensor_samples)
    dataloader = DataLoader(
        torch_dataset, 
        batch_size=training_params['batch_size'], 
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    
    # SANITY CHECK: does the model with the lowest log probability correspond to the true ordering?
    logging.info("Checking whether model with true ordering has highest log likelihood...")
    perms = list(permutations(range(num_covariates)))
    results = {}

    for perm in perms:
        model = create_oslow_model_with_ordering(perm).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.5)
        
        for epoch in range(30):  # Train for 30 epochs
            epoch_log_probs = []
            for batch, in dataloader:
                batch = batch.to(device)
                log_prob = model.log_prob(batch).mean()
                loss = -log_prob
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_log_probs.append(log_prob.item())

            results[perm] = epoch_log_probs[-1]

    # Log results
    logging.info("Sanity check results:")
    for perm, log_prob in sorted(results.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"   Ordering {perm}: Final log likelihood = {log_prob}")
    
    logging.info(f"True ordering: {tuple(true_ordering)}")

    logging.info("\nStarting recursive ordering algorithm...")
    discovered_ordering = determine_ordering(
        remaining_covariates=list(range(num_covariates)),
        dataloader=dataloader,
        model_params=model_params,
        training_params=training_params,
        true_ordering=true_ordering,
        num_total_covariates=num_covariates
    )
    
    logging.info(f"True ordering: {true_ordering}")
    logging.info(f"Discovered ordering: {discovered_ordering}")
    
    return {
        "true_ordering": true_ordering,
        "discovered_ordering": discovered_ordering
    }

# Main script
if __name__ == "__main__":
    num_covariates = 3
    num_samples = 30000
    true_ordering = [1, 2, 0]

    graph_generator = GraphGenerator(
        num_nodes=num_covariates,
        seed=0,
        graph_type="full",
        enforce_ordering=true_ordering,
    )

    gaussian_noise_generator = RandomGenerator('normal', seed=10, loc=0, scale=1)
    laplace_noise_generator = RandomGenerator('laplace', seed=10, loc=0, scale=1)
    link_generator = RandomGenerator('uniform', seed=110, low=1, high=1)

    datasets = {
        "sinusoidal": AffineParametericDataset(
            num_samples=num_samples,
            graph_generator=graph_generator,
            noise_generator=gaussian_noise_generator,
            link_generator=link_generator,
            link="sinusoid",
            perform_normalization=False,
        ),
        "laplace_linear": AffineParametericDataset(
            num_samples=num_samples,
            graph_generator=graph_generator,
            noise_generator=laplace_noise_generator,
            link_generator=link_generator,
            link="linear",
            perform_normalization=False,
            additive=True,
        ),
        "nonparametric_affine": AffineNonParametericDataset(
            num_samples=1000,
            graph_generator=graph_generator,
            noise_generator=gaussian_noise_generator,
            invertibility_coefficient=0.0,
            perform_normalization=False,
            additive=False,
        ),
        "nonparametric_additive": AffineNonParametericDataset(
            num_samples=1000,
            graph_generator=graph_generator,
            noise_generator=gaussian_noise_generator,
            invertibility_coefficient=0.0,
            perform_normalization=False,
            additive=True,
        ),
        "nonparametric_almost_invertible": AffineNonParametericDataset(
            num_samples=1000,
            graph_generator=graph_generator,
            noise_generator=gaussian_noise_generator,
            invertibility_coefficient=1.0,
            perform_normalization=False,
            additive=False,
        )
    }

    logging.info("Datasets generated:")
    for name in datasets.keys():
        logging.info(f"- {name}")

    test_results = {}
    for dataset_name, dataset in datasets.items():
        test_results[dataset_name] = run_single_test(dataset_name, dataset, true_ordering, num_covariates)

    logging.info("\nTest Results Summary:")
    for dataset_name, result in test_results.items():
        logging.info(f"{dataset_name}:")
        logging.info(f"  True ordering: {result['true_ordering']}")
        logging.info(f"  Discovered ordering: {result['discovered_ordering']}")