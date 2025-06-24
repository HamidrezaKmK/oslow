import torch
from typing import List, Literal
# from training.utils import listperm2matperm

def uniform_gamma_init_func(
    in_features: int,
    mode: Literal['permutation-matrix', 'list'] = 'permutation-matrix',
):
    """
    Initialize the gamma parameter with identical values. 
    This is equivalent to the uniform distribution over the set of permutations.
    
    Args:
        in_features: The number of features in the input.
        mode: Whether the parameterization uses a permutation matrix or a list of size `in_features` of learnable parameters.
    Returns:
        A tensor of shape (in_features, in_features) if mode is 'permutation-matrix' and (in_features, ) otherwise.     
    """
    if mode == 'permutation-matrix':
        repeat = 2
    elif mode == 'list':
        repeat = 1
    else:
        raise ValueError(f"Expected mode to be 'permutation-matrix' or 'list', got {mode}")
    return torch.ones(*[in_features for _ in range(repeat)])

def normal_gamma_init_func(
    in_features: int,
    mode: Literal['permutation-matrix', 'list'] = 'permutation-matrix',
):
    """
    Initialize the gamma parameter with samples from an isotropic Gaussian distribution.
    
    Args:
        in_features: The number of features in the input.
        mode: Whether the parameterization uses a permutation matrix or a list of size `in_features` of learnable parameters.
    Returns:
        A tensor of shape (in_features, in_features) if mode is 'permutation-matrix' and (in_features, ) otherwise.     
    """
    if mode == 'permutation-matrix':
        repeat = 2
    elif mode == 'list':
        repeat = 1
    else:
        raise ValueError(f"Expected mode to be 'permutation-matrix' or 'list', got {mode}")
    return torch.randn(*[in_features for _ in range(repeat)])

