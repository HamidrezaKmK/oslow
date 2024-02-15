import torch
import typing as th
import numpy as np
import functools

# most of the codes are humbly borrowed/adapted from
# https://github.com/sharpenb/Differentiable-DAG-Sampling/tree/44f96769a729efc99bdd16c9b00deee4077a76b2


def translate_idx_ordering(
    idx: th.Union[th.List[int], th.List[th.List[int]], np.array]
):
    if isinstance(idx[0], list):
        return [translate_idx_ordering(i) for i in idx]
    else:
        return [j for i, j in sorted(zip(idx, range(len(idx))))]


def turn_into_matrix(permutation: torch.IntTensor):
    return (
        torch.eye(permutation.shape[0])
        .to(permutation.device)[permutation]
        .to(permutation.device)
    )


@functools.wraps(torch.rand)
def sample_gumbel_noise(*args, eps=1e-20, std=1, **kwargs):
    """Samples arbitrary-shaped standard gumbel variables.
    Args:
        shape: list of integers
        eps: float, for numerical stability
    Returns:
        A sample of standard Gumbel random variables
    """

    u = torch.rand(*args, **kwargs).float()
    return -torch.log(-torch.log(u + eps) + eps) * std


def gumbel_log_prob(gumbel_noise: torch.Tensor) -> torch.Tensor:
    """Computes the log-probability of a sample of Gumbel random variables.
    Args:
        gumbel_noise: a sample of Gumbel random variables
    Returns:
        The log-probability of the sample
    """
    return (-gumbel_noise - torch.exp(-gumbel_noise)).sum(dim=[-1, -2])


def sinkhorn(log_alpha, num_iters=20):
    """Performs incomplete Sinkhorn normalization to log_alpha.
    By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the succesive row and column
    normalization.
        -To ensure positivity, the effective input to sinkhorn has to be
        exp(log_alpha) (elementwise).
        -However, for stability, sinkhorn works in the log-space. It is only at
        return time that entries are exponentiated.

    [1] https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-21/issue-2/\
        Concerning-nonnegative\-matrices-and-doubly-stochastic-matrices

    Args:
        log_alpha: 2D tensor (a matrix of shape [N, N])
            or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
        num_iters: number of sinkhorn iterations (in practice, as little as 20
            iterations are needed to achieve decent convergence for N~100)
    Returns:
        A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
            converted to 3D tensors with batch_size equals to 1)
    """
    n = log_alpha.size()[1]
    log_alpha = log_alpha.reshape(-1, n, n)
    for _ in range(num_iters):
        log_alpha = log_alpha - (
            torch.logsumexp(log_alpha, dim=2, keepdim=True)
        ).reshape(-1, n, 1)
        log_alpha = log_alpha - (
            torch.logsumexp(log_alpha, dim=1, keepdim=True)
        ).reshape(-1, 1, n)

    results = torch.exp(log_alpha)
    return results


def is_doubly_stochastic(mat, threshold: th.Optional[float] = 1e-4) -> torch.Tensor:
    """
    Checks if a matrix is doubly stochastic (given a threshold).
    Args:
        mat: 2D tensor (a matrix of shape [N, N])
            or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
        threshold: float, the threshold for the check to pass
            if None, the maximum difference between the sum of rows and columns and 1 is returned

    Returns:
        Either a boolean tensor of shape [batch_size] (if threshold is not None)
        or a float tensor of shape [batch_size] (if threshold is None)
    """
    if threshold is None:
        # return the maximum difference between the sum of rows and columns and 1
        return torch.maximum(
            (mat.sum(-1) - 1).abs().max(-1).values,
            (mat.sum(-2) - 1).abs().max(-1).values,
        )
    return ((mat.sum(-1) - 1).abs().max(-1).values < threshold) & (
        (mat.sum(-2) - 1).abs().max(-1).values < threshold
    )


def is_permutation(mat, threshold: th.Optional[float] = 1e-4):
    """
    Checks if a matrix is a permutation matrix (given a threshold).
    By definition, a permutation matrix is a doubly stochastic matrix with
    exactly one 1 per row and column.

    To check how close a matrix is to being a permutation matrix, we check
    how close all of its entries are to 0 or 1. The maximum distance is then used
    as a measure of how close the matrix is to being a permutation matrix.

    Args:
        mat: 2D tensor (a matrix of shape [N, N])
            or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
        threshold: float, the threshold for the check to pass
            if None, the maximum distance between values and 0 or 1 is returned

    Returns:
        Either a boolean tensor of shape [batch_size] (if threshold is not None)
        or a float tensor of shape [batch_size] (if threshold is None)
    """
    distance = torch.minimum((mat - 1).abs(), mat.abs())
    results = distance.max(-1).values.max(-1).values
    return results if threshold is None else results < threshold


def is_between_zero_one(mat, threshold: th.Optional[float] = 1e-4):
    """
    Checks if a matrix is between 0 and 1 (given a threshold). The difference between
    this funciton and is_permutation is that, here we only care how much of the values
    are not in between 0 and 1, while in is_permutation we only care that the values are
    close to either 0 or 1.

    is_permutation might be true (having small value)
    Args:
        mat: 2D tensor (a matrix of shape [N, N])
            or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
        threshold: float, the threshold for the check to pass
            if None, the maximum distance between values and 0 or 1 is returned
    """
    # zero out all the numbers except the ones that are not between 0 and 1
    mat = torch.where((mat >= 0) & (mat <= 1.0), torch.zeros_like(mat), mat)
    distance = torch.minimum((mat - 1).abs(), mat.abs())
    results = distance.max(-1).values.max(-1).values
    return results if threshold is None else results < threshold


def evaluate_permutations(
    mat, threshold: th.Optional[float] = 1e-4, reduce: bool = True
):
    """
    Evaluates a matrix of permutations (or a batch of matrices of permutations)

    Args:
        mat: 2D tensor (a matrix of shape [N, N]) or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
        threshold: float, the threshold for the check to pass (if None, only the distances are returned)
        reduce: whether to reduce the results into an average (boolean tensors are reduced to a proportion of True values)

    Returns:
        A dictionary with the following keys:
            doubly_stochastic_distance: the maximum difference between the sum of rows and columns and 1
            permutation_distance: the maximum distance between values and 0 or 1
            between_zero_one_distance: the maximum distance between values and 0 or 1
            is_doubly_stochastic: whether the matrix is doubly stochastic (if threshold is not None)
            is_permutation: whether the matrix is a permutation matrix (if threshold is not None)
            is_between_zero_one: whether the matrix is between 0 and 1 (if threshold is not None)
    """
    results = dict(
        doubly_stochastic_distance=is_doubly_stochastic(mat, threshold=None),
        permutation_distance=is_permutation(mat, threshold=None),
        between_zero_one_distance=is_between_zero_one(mat, threshold=None),
    )
    if threshold is not None:
        results.update(
            {
                f'is_{k.replace("_distance", "")}': v < threshold
                for k, v in results.items()
            }
        )
    if reduce:
        results = {k: v.float().mean() for k, v in results.items()}
    return results


def listperm2matperm(
    listperm: th.Union[torch.Tensor, th.List[int]], device=None, dtype=None
):
    """Converts a batch of permutations to its matricial form.
    Args:
      listperm: 2D tensor of permutations of shape [batch_size, n_objects] so that
        listperm[n] is a permutation of range(n_objects).
      device: device to place the output tensor on. (default: None)
      dtype: dtype of the output tensor. (default: None)
    Returns:
      a 3D tensor of permutations matperm of
        shape = [batch_size, n_objects, n_objects] so that matperm[n, :, :] is a
        permutation of the identity matrix, with matperm[n, i, listperm[n,i]] = 1
    """
    listperm = (
        torch.as_tensor(listperm, device=device)
        if not isinstance(listperm, torch.Tensor)
        else listperm
    )
    return torch.eye(listperm.shape[-1], device=device)[listperm.long()].to(
        device=device, dtype=dtype
    )


def hungarian(matrix_batch):
    """Solves a matching problem using the Hungarian algorithm.

    This is a wrapper for the scipy.optimize.linear_sum_assignment function. It
    solves the optimization problem max_P sum_i,j M_i,j P_i,j with P a
    permutation matrix. Notice the negative sign; the reason, the original
    function solves a minimization problem.

    Args:
        matrix_batch: A 3D tensor (a batch of matrices) with
            shape = [batch_size, N, N]. If 2D, the input is reshaped to 3D with
            batch_size = 1.
    Returns:
        listperms, a 2D integer tensor of permutations with shape [batch_size, N]
            so that listperms[n, :] is the permutation of range(N) that solves the
            problem  max_P sum_i,j M_i,j P_i,j with M = matrix_batch[n, :, :].
    """
    # keep the import here to avoid unnecessary dependency in the rest of the code
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    # perform the hungarian algorithm on the cpu
    device = matrix_batch.device
    matrix_batch = matrix_batch.detach().cpu().numpy()
    if matrix_batch.ndim == 2:
        matrix_batch = np.reshape(
            matrix_batch, [1, matrix_batch.shape[0], matrix_batch.shape[1]]
        )
    sol = np.zeros((matrix_batch.shape[0], matrix_batch.shape[1]), dtype=np.int32)
    for i in range(matrix_batch.shape[0]):
        sol[i, :] = linear_sum_assignment(-matrix_batch[i, :])[1].astype(np.int32)
    return torch.from_numpy(sol).to(device).detach()


def all_permutations(n):
    """Returns all permutations of range(n)"""
    if n == 1:
        return [[0]]
    else:
        perms = []
        for perm in all_permutations(n - 1):
            for i in range(n):
                perms.append(perm[:i] + [n - 1] + perm[i:])
        return perms


def abbriviate_permutation(permutation_list: th.Iterable[int]) -> str:
    """
    Loop over the permutation_list (a list of unique integers) and find all consecutive sequences
    of ascending/descending numbers and replace them with the first and last number in
    the sequence. Numbers are expected to be between 0 and len(permutation_list) - 1.

    For example: [0, 1, 2, 3, 4] -> ["0-4"] and [4, 3, 2, 1, 0] -> ["4-0"]
        or [0, 1, 2, 3, 10, 9, 8, 4, 5, 7, 6] -> ["0-3", "10-8", "4-5", "7-6"]

    This is useful for plotting the permutation_list of the permutation matrix in a more compact way.

    Args:
        permutation_list (th.Iterable[int]): A list of unique integers between 0 and len(permutation_list) - 1

    Returns:
       an abbreviated string representation of the permutation_list
    """
    if len(permutation_list) < 2:
        return permutation_list
    permutation_list, results = list(permutation_list), []
    ascending: bool = permutation_list[0] < permutation_list[1]
    start, start_idx = permutation_list[0], 0
    for i in range(1, len(permutation_list)):
        if ascending and permutation_list[i] == permutation_list[i - 1] + 1:
            continue
        elif not ascending and permutation_list[i] == permutation_list[i - 1] - 1:
            continue
        else:
            if start_idx == i - 1:
                results.append(f"{start}")
            else:
                results.append(f"{start}-{permutation_list[i-1]}")
            start, start_idx = permutation_list[i], i
            ascending = (
                permutation_list[i] < permutation_list[i + 1]
                if i < len(permutation_list) - 1
                else ascending
            )

    if start_idx < len(permutation_list) - 1:
        results.append(f"{start}-{permutation_list[-1]}")
    else:
        results.append(f"{start}")
    return f'[{",".join(results)}]'


def generate_permutations(n: int, num_samples: int = 1, return_matrix: bool = True):
    """
    Generates num_samples random permutation (matrices)

    Args:
        n (int): the number of elements to permute
        num_samples (int): the number of permutations to generate
        return_matrix (bool): whether to return the permutations as a matrix or as a list of orders

    Returns:
        a 3D tensor of shape [num_samples, n, n] or a list of lists of length num_samples
    """

    results = torch.empty(num_samples, n).long()
    num_unique = 0
    while num_unique < num_samples:
        all_perms = torch.cat(
            [results[:num_unique], torch.randperm(n).reshape(1, -1)], dim=0
        ).unique(dim=0)
        results[: len(all_perms)] = all_perms
        num_unique = len(all_perms)

    if return_matrix:
        return listperm2matperm(results)
    return results
