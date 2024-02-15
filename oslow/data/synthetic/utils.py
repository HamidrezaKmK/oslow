import numpy as np
from typing import Optional, Tuple, Union, Literal
from functools import partial
from sklearn.metrics.pairwise import rbf_kernel


class RandomGenerator:
    """
    Generalized reproducible random generator capable of generating different types of random variables.

    For example, given noise_type == 'laplace', this generator will generate samples from a Laplace distribution.

    Instantiation examples:
        generator = RandomGenerator('normal', loc=0, scale=1) # generates samples from a standard normal distribution
        generator = RandomGenerator('normal', loc=0, scale=1, seed=42) # generates samples from a standard normal distribution with seed 42
        generator = RandomGenerator('laplace', loc=0, scale=1) # generates samples from a Laplace distribution with mean 0 and scale 1
        generator = RandomGenerator('uniform', low=0, high=1) # generates samples from a uniform distribution in the range [0, 1)

    Getting samples:
        samples = generator(size=(100, 100)) # generates a 100 by 100 matrix of samples from the specified distribution
    """

    def __init__(self, noise_type: str, seed: Optional[int], *args, **kwargs):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        if not hasattr(rng, noise_type):
            raise ValueError(f"Unknown noise type {noise_type}")

        self.rng = partial(getattr(rng, noise_type), *args, **kwargs)

    def __call__(self, size: Union[int, Tuple[int, ...]]):
        """
        The class is callable and returns a tensor of size `size` with the specified noise type.
        Each element in the tensor is sampled independently from the noise distribution.
        """
        return self.rng(size=size)


def softplus(x, threshold: float = 20.0):
    # A numerically stable implementation of softplus
    return np.where(x > threshold, x, np.log(1 + np.exp(x)))


def standardize(x):
    # Standardize the data to have zero mean and unit variance and also account for the case where the variance is zero
    return (x - np.mean(x)) / (np.std(x) if np.std(x) > 1e-3 else 1e-3)


def perform_post_non_linear_transform(
    x,
    type: Literal["exp", "softplus", "x_plus_sin", "nonparametric", "sinusoid"],
    additional_params: Optional[dict] = None,
):
    """
    Perform a post non-linear transform on the data.
    """
    if type == "exp":
        return np.exp(x)
    elif type == "softplus":
        return np.where(x > 20.0, x, np.log(1 + np.exp(x)))
    elif type == "x_plus_sin" or type == "sinusoid":
        return x + np.sin(x)
    elif type == "nonparametric":
        additional_params = additional_params or {}
        kernel = rbf_kernel(x.reshape(-1, 1), **additional_params)
        return np.matmul(kernel, x.reshape(-1, 1))
    else:
        raise Exception(f"Unknown post non-linear transform {type}")
