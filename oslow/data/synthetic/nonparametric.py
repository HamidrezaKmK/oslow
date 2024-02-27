"""
This file contains an affine/additive non parametric SCM generator
that can generte datasets with parametric assumptions and is appropriate for testing out the affine models.
"""

from typing import Optional, Literal, List
import networkx as nx
import numpy as np
import math
from .utils import RandomGenerator
from oslow.data.base_dataset import OCDDataset
import pandas as pd
from oslow.data.synthetic.graph_generator import GraphGenerator
from sklearn.metrics.pairwise import rbf_kernel
from .utils import perform_post_non_linear_transform, softplus, standardize


class AffineNonParametericDataset(OCDDataset):
    """
    This is an SCM generator that generates SCMs with the following function formulas:

    x_i = t(Pa_i) + s(Pa_i) * Noise(**Parameters)

    where t and s are sampled from a Gaussian process with a squared exponential kernel and
    Noise is a noise function generated.
    """

    def __init__(
        self,
        num_samples: int,
        graph_generator: GraphGenerator,
        noise_generator: RandomGenerator,
        *args,
        s_rbf_kernel_gamma: float = 1.0,
        t_rbf_kernel_gamma: float = 1.0,
        invertibility_coefficient: float = 0.0,
        perform_normalization: bool = True,
        additive: bool = False,
        post_non_linear_transform: Optional[
            Literal["exp", "softplus", "x_plus_sin", "sinusoid", "nonparametric"]
        ] = None,
        **kwargs,
    ):
        """
        Args:
            num_samples: number of samples to generate
            graph_generator: An object of type GraphGenerator that generates the DAG
            noise_generator: An object of type RandomGenerator that generates the parameters for the noise function
            link_generator: An object of type RandomGenerator that generates the parameters for the s and t functions
            link: a string representing the link function to use
            perform_normalization: whether to normalize the data after generating the column, this is done for numerical stability
            additive: whether to use an additive noise model or not, for an additive model the noise does not get modulated
        """
        graph: nx.DiGraph = graph_generator.generate_dag()
        topological_order = list(nx.topological_sort(graph))
        # create an empty pandas dataframe with columns equal to the graph.nodes and rows equal to num_samples and initialize it with zeros
        dset = pd.DataFrame(
            0, index=range(num_samples), columns=sorted(list(graph.nodes()))
        )
        # iterate over the nodes in the topological order
        for v in topological_order:
            noises = noise_generator(num_samples)
            # get incoming nodes to v in graph
            parents = list(graph.predecessors(v))

            if len(parents) == 0:
                s = np.random.normal(0, s_rbf_kernel_gamma, num_samples)
                t = np.random.normal(0, t_rbf_kernel_gamma, num_samples)
            else:
                parent_values = dset[parents].values
                s_kernel = rbf_kernel(parent_values, gamma=s_rbf_kernel_gamma)
                t_kernel = rbf_kernel(parent_values, gamma=t_rbf_kernel_gamma)
                s = np.random.multivariate_normal(np.zeros(num_samples), s_kernel)
                # perform a softplus on s to ensure positivity
                s = softplus(s)
                # perform a transformation to increase the probability of t being non-constant w.r.t. each parent
                t = np.random.multivariate_normal(np.zeros(num_samples), t_kernel)
                t = t + np.sum(parent_values, axis=1) * invertibility_coefficient

            if additive:
                x = t + noises
            else:
                x = t + s * noises

            # normalize the data if specified
            if perform_normalization:
                x = standardize(x)

            if post_non_linear_transform is not None:
                x = perform_post_non_linear_transform(x, type=post_non_linear_transform)

            dset[v] = x.reshape(-1, 1)
        if not "name" in kwargs:
            kwargs["name"] = f"AffineNonParametericDataset"
        super().__init__(dset, graph, *args, **kwargs)
