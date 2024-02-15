"""
Given a causal ordering, we can prune the causal ordering to remove
any causal links that are not necessary for the causal graph.

This decoupled procedure will give us a full algorithm to obtain the 
causal underlying structure from a dataset.
"""

import numpy as np
import typing as th
import pandas as pd
import sklearn as sk
from tqdm import tqdm
import scipy as sp

# define a python enum for the different pruning methods


class PruningMethod:
    CONDITIONAL_INDEPENDENCE_TESTING = 0
    SPARSE_PREDICTION = 1
    LEAVE_ONE_OUT = 2


class IndependceTestingMethod:
    MUTUAL_INFORMATION = 0
    CHI_SQUARED = 1


def conditional_independence_test(
    x: np.array,
    y: np.array,
    z: th.Optional[np.array] = None,
    threshold: float = 0.05,
    sample_threshold: int = 20,
    categorical: bool = True,
    independence_test: IndependceTestingMethod = IndependceTestingMethod.CHI_SQUARED,
) -> bool:
    """
    This algorithm takes in two variables and a conditioning set and performs a conditional independence test.
    The test is performed by computing the mutual information between the two variables and conditioning on the conditioning set.
    We will first group the conditioning set by unique values and then perform the independence test on each group.
    To do so, on each group we calculate a score and then average the scores weighted by the number of samples in each group.
    For the groups that have less than the sample threshold, we ignore them.
    If the average score is less than the threshold then we return True, otherwise we return False.

    Args:
        x (np.array): first (set) of variables
        y (np.array): second variable
        z (np.array): conditioning set (optional) if set to None then it is a simple independence test
        threshold (float): threshold for the p-value
        sample_threshold (int): minimum number of samples required for a group to be considered

    Returns:
        bool: True if the variables are conditionally independent, False otherwise
    """

    if categorical:
        # get the number of unique values in x and y
        x_unique = np.unique(x)
        y_unique = np.unique(y)

    def calculate_score(x, y):
        if independence_test == IndependceTestingMethod.MUTUAL_INFORMATION:
            return sk.metrics.mutual_info_score(x, y)
        elif independence_test == IndependceTestingMethod.CHI_SQUARED:
            tab = pd.crosstab(x, y)
            p_value = sp.stats.chi2_contingency(tab)[1]
            return p_value
        else:
            raise ValueError("Invalid independence test method")

    def compare_with_threshod(score, threshold):
        if independence_test == IndependceTestingMethod.MUTUAL_INFORMATION:
            return score < threshold
        else:
            return score > threshold

    if z is None:
        return compare_with_threshod(calculate_score(x, y), threshold)

    # group rows of z by unique values and do an independence test on each group
    z_unique, z_unique_idx, z_unique_inv, z_unique_counts = np.unique(
        z, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    sm = 0
    count = 0
    for i in range(len(z_unique_counts)):
        if z_unique_counts[i] < sample_threshold:
            continue
        xs = x[z_unique_inv == i]
        ys = y[z_unique_inv == i]
        # check positivity
        positivity = True
        if categorical:
            if len(np.unique(xs)) != len(x_unique) or len(np.unique(ys)) != len(y_unique):
                positivity = False
        if not positivity:
            continue
        # calculate score
        score = calculate_score(xs, ys)
        count += z_unique_counts[i]
        sm += z_unique_counts[i] * score
    if count == 0:
        return True
    # compare the average score with threshold
    return compare_with_threshod(sm / count, threshold)


def prune_cit(data: pd.DataFrame, y: int, x: int, all_prec: th.List[int], *args, **kwargs) -> bool:
    """
    Prune the edge from y->x using conditional independence testing
    It is assumed that x depends on a subset of all_prec and we want to make sure
    if y is essential to the causal relationship between x and all its parents or not.
    If y is a non-essential node, then we can simply remove it.

    The way this is done is that the set all_prec \ {y} is used to condition on
    and a conditional independence test is performed between x and y.
    If the p-value is above the threshold, then y is essential and we cannot remove it.
    Otherwise, we can remove it.

    Args:
        data (pd.DataFrame): the dataset
        y (int): the index of the parent node corresponding the the same index in the columns of data
        x (int): the index of the child node corresponding the the same index in the columns of data
        all_prec (list): list of indices of the nodes that are parents of x (y included)

    Returns:
        bool: True if y can be removed and False otherwise
    """
    # do a conditional independence test between x and y conditioned on all_prec
    # if the p-value is above the threshold, then y is not a parent of x
    # otherwise, y is a parent of x

    cp_all_prec = all_prec.copy()
    cp_all_prec.remove(y)
    x = data.iloc[:, x].values
    y = data.iloc[:, y].values
    if len(cp_all_prec) == 0:
        return conditional_independence_test(x, y, None, *args, **kwargs)
    z = data.iloc[:, cp_all_prec].values

    return conditional_independence_test(x, y, z, *args, **kwargs)


def prune_sp(data: pd.DataFrame, x: int, all_prec: th.List[int]) -> th.List[int]:
    """
    Use a sparse prediction model to prune the edges from the parents of x

    Using the columns in all_prec we will train a model to predict x.
    Furthermore, using sparse techniques such as lasso regularization, we will
    find the subset of all_prec that are most important for predicting x.

    Args:
        data (pd.DataFrame): the dataset
        x (int): the index of the child node corresponding the the same index in the columns of data
        all_prec (list): list of indices of the nodes that are parents of x

    Returns:
        list: list of indices of the nodes that are essential parents of x
    """

    raise NotImplementedError("Sparse regression not implemented yet")


def prune_loo(data: pd.DataFrame, y: int, x: int, all_prec: th.List[int]) -> bool:
    raise NotImplementedError("Leave one out not implemented yet")


def create_dag_from_ordering(ordering: th.List[int]) -> np.array:
    """
    Create a DAG from a topological ordering of the nodes
    """
    n = len(ordering)
    dag = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dag[ordering[i], ordering[j]] = 1
    return dag


def prune(
    ordering: th.List[int],
    data: pd.DataFrame,
    method: PruningMethod,
    interventional_column: th.Optional[int] = None,
    dag: th.Optional[np.array] = None,
    verbose: int = 0,
    method_params: th.Optional[dict] = None,
) -> np.array:
    """
    This function prunes a given DAG or a given ordering with a corresponding DAG tournament.
    The pruning method is specified by the method parameter and the parameters for the method
    are specified by the method_params parameter.

    The method in general iterates over the ordering and considers the parents of each node
    and then using a pruning technique, it decides if the parents are essential or not.
    If they are not essential, then the edge is removed.

    Args:
        ordering (list): causal ordering of the nodes
        dag (np.array): adjacency matrix of the DAG if initialized
        data (pd.DataFrame): data
        method (PruningMethod): method to use for pruning
        method_params (dict): parameters for the pruning method
        verbose (int): 0 for no progress bar, 1 for progress bar, 2 for printing logs

    Returns:
        np.array: pruned dag


    Prune the ordering to remove any causal links that are not necessary for the causal graph
    """
    method_params = method_params or {}

    # If no dag is given then we assume that the ordering is a topological ordering
    # and we create a dag from it
    dag = create_dag_from_ordering(ordering) if dag is None else dag

    # take a copy of dag and ordering
    dag = dag.copy()
    ordering = ordering.copy()

    # Set a progree bar if verbose is 1
    if verbose == 1:
        my_iterable = tqdm(range(dag.shape[0]))
    else:
        my_iterable = range(dag.shape[0])

    for i in my_iterable:
        # If verbose is 2 then print the node that is being pruned
        if verbose == 2:
            print("Pruning parents of node: ", ordering[i])

        x = ordering[i]

        # Ignore the interventional column
        if interventional_column is not None and x == interventional_column:
            continue

        # Create a list of all the parents of x in all_prec
        all_prec = []
        for j in range(i):
            if dag[ordering[j], ordering[i]] == 1:
                all_prec.append(ordering[j])
        # Get a list of current parents of x
        if method == PruningMethod.SPARSE_PREDICTION:
            # Get all the essential parents via sparse regression
            essential_parents = prune_sp(data, x, all_prec, **method_params)
            for y in all_prec:
                if y not in essential_parents:
                    dag[y, x] = 0
            continue

        # The other methods are based on pruning each edge one by one
        if "n_repeat" not in method_params:
            method_params["n_repeat"] = 1
        n_repeat = method_params["n_repeat"]
        method_params.pop("n_repeat", None)

        # The current active parent
        current_parents = all_prec.copy()
        for _ in range(n_repeat):
            for y in all_prec:
                # check if the edge is necessary
                do_prune = False
                if method == PruningMethod.CONDITIONAL_INDEPENDENCE_TESTING:
                    do_prune = prune_cit(data, y, x, current_parents, **method_params)
                elif method == PruningMethod.LEAVE_ONE_OUT:
                    do_prune = prune_loo(data, y, x, current_parents, **method_params)
                else:
                    raise ValueError("Invalid pruning method")

                if do_prune:
                    # print the pruning if verbose is 2
                    if verbose > 1:
                        print("Pruning {} -> {}".format(y, x))
                    current_parents.remove(y)
                    dag[y, x] = 0
    return dag
