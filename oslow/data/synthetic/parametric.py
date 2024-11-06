"""
This file contains a fast parametric SCM generator
that can generte datasets with parametric assumptions and is appropriate for testing out the affine models.
"""

from oslow.data.synthetic.graph_generator import GraphGenerator
from typing import Optional, Literal, List
import networkx as nx
import numpy
import numpy as np
import math
from .utils import RandomGenerator
from oslow.data.base_dataset import OCDDataset
import pandas as pd
from .utils import perform_post_non_linear_transform, softplus, standardize


class AffineParametericDataset(OCDDataset):
    """
    This is an SCM generator that generates SCMs with the following function formulas:

    x_i = t(Pa_i) + s(Pa_i) * Noise(**Parameters)

    where t is a non-linear invertible function and
    s is a positive function, and Noise is a noise function generated
    from a noise type and its parameters that can be specified in the constructor.

    The following functions are used to compute t and s:
    s_inp(x_1, x_2, ..., x_k) = s_1 * x_1 + s_2 * x_2 + ... + s_k * x_k + s_0
    t_inp(x_1, x_2, ..., x_k) = t_1 * x_1 + t_2 * x_2 + ... + t_k * x_k + t_0
    where s_i and t_i are generated independently from a link_generator.

    Finally, using the s_function and t_function, we get the actual s and t functions:

    s = softplus(s_inp), t = t_function(t_inp)

    where t_function is set according to the link parameter in the constructor.

    t_function(x) = sin(x) + x if link == "sinusoid"
    t_function(x) = x**3 if link == "cubic"
    t_function(x) = x if link == "linear"
    x^2 if link == "square" 
    |X| if link == "absolute"
    """

    def __init__(
        self,
        num_samples: int,
        noise_generator: RandomGenerator,
        link_generator: RandomGenerator,
        graph_generator: GraphGenerator,
        *args,
        link: Literal["sinusoid", "cubic", "linear", "square", "absolute", "sigmoid"] = "sinusoid",
        perform_normalization: bool = True,
        additive: bool = False,
        post_non_linear_transform: Optional[
            Literal["exp", "softplus", "x_plus_sin", "nonparametric"]
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

            s_coeffs = link_generator(len(parents) + 1)
            t_coeffs = link_generator(len(parents) + 1)
            if len(parents) == 0:
                s_pre = np.ones(num_samples) * s_coeffs[-1]
                t_pre = np.ones(num_samples) * t_coeffs[-1]
            else:
                s_pre = (
                    np.einsum("i,ji->j", s_coeffs[:-1], dset[parents].values)
                    + s_coeffs[-1]
                )
                t_pre = (
                    np.einsum("i,ji->j", t_coeffs[:-1], dset[parents].values)
                    + t_coeffs[-1]
                )
            # pass s_pre through a softplus
            s = softplus(s_pre)
            if link == "sinusoid":
                t = np.sin(t_pre) + t_pre
            elif link == "cubic":
                t = t_pre**3
            elif link == "linear":
                t = t_pre
            elif link == "square":
                t = t_pre**2
            elif link == "absolute":
                t = np.abs(t_pre)
            elif link == "sigmoid":
                t = 1 / (1 + np.exp(-t_pre))
            else:
                raise Exception(f"Link {link} not supported")
            x = t + (s * noises if not additive else noises)

            if perform_normalization: 
                # NOTE: be sure that standardize is not called in OCDDataset if we perform normalization here 
                x = standardize(x)

            if post_non_linear_transform is not None:
                x = perform_post_non_linear_transform(x, type=post_non_linear_transform)

            dset[v] = x.reshape(-1, 1)

        if not "name" in kwargs:
            kwargs["name"] = f"AffineParametericDataset"

        super().__init__(dset, graph, *args, **kwargs)  
        # NOTE: be sure that standardize is not called in OCDDataset if we perform normalization earlier
