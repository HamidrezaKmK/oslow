"""
Structural Intervention Distance (SID) for Evaluating Causal Graphs

This module implements the SID metric as described in:
Peters, J., & Bühlmann, P. (2014). Structural intervention distance (SID) for evaluating causal graphs.

The SID measures the structural differences between two directed acyclic graphs (DAGs)
in terms of their corresponding causal inference statements and intervention distributions.

Translated from the original R implementation by Jonas Peters.
"""

import numpy as np
from typing import Tuple, List, Optional


def compute_path_matrix(G: np.ndarray, sparse: bool = False) -> np.ndarray:
    """
    Computes a path matrix for which entry (i,j) being one means that there is a
    directed path from i to j. The diagonal will also be one.

    Args:
        G: Adjacency matrix of a DAG
        sparse: Whether to use sparse matrix operations (not implemented in this version)

    Returns:
        Path matrix with reachability information
    """
    p = G.shape[0]

    if p > 3000 and not sparse:
        print("Warning: Maybe you should use the sparse version to increase speed")

    # Initialize path matrix with identity + adjacency matrix
    path_matrix = np.eye(p) + G

    # Use matrix multiplication to compute transitive closure
    k = int(np.ceil(np.log2(p)))
    for _ in range(k):
        path_matrix = np.dot(path_matrix, path_matrix)

    # Convert to boolean matrix
    path_matrix = path_matrix > 0

    return path_matrix


def compute_path_matrix2(
    G: np.ndarray, cond_set: List[int], path_matrix1: np.ndarray, sparse: bool = False
) -> np.ndarray:
    """
    Similar to compute_path_matrix but removes all edges leaving cond_set.
    If cond_set is empty, it just returns path_matrix1.

    Args:
        G: Adjacency matrix of a DAG
        cond_set: A list of node indices (conditioning set)
        path_matrix1: The original path matrix (for the case when cond_set is empty)
        sparse: Whether to use sparse matrix operations (not implemented in this version)

    Returns:
        Modified path matrix with reachability information
    """
    p = G.shape[0]

    if len(cond_set) > 0:
        # Create a copy to avoid modifying the original
        G_modified = G.copy()

        # Remove all edges that leave condSet
        for node in cond_set:
            G_modified[node, :] = 0

        # Initialize path matrix with identity + modified adjacency matrix
        path_matrix2 = np.eye(p) + G_modified

        # Use matrix multiplication to compute transitive closure
        k = int(np.ceil(np.log2(p)))
        for _ in range(k):
            path_matrix2 = np.dot(path_matrix2, path_matrix2)

        # Convert to boolean matrix
        path_matrix2 = path_matrix2 > 0
    else:
        path_matrix2 = path_matrix1

    return path_matrix2


def find_reachable_on_non_directed_path(
    G: np.ndarray,
    i: int,
    cond_set: List[int],
    path_matrix: np.ndarray,
    path_matrix2: Optional[np.ndarray] = None,
    sparse: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """
    Find all nodes that can be reached from i on a non-directed path that is not blocked
    by cond_set. This implements the rondp function from the R code.

    Args:
        G: Adjacency matrix of a DAG
        i: Source node index
        cond_set: Conditioning set (nodes that block paths)
        path_matrix: Original path matrix
        path_matrix2: Modified path matrix without arrows leaving cond_set
        sparse: Whether to use sparse matrix operations

    Returns:
        Tuple containing:
        - Array indicating which nodes are reachable on non-directed paths
        - Time spent computing path_matrix2
        - Time spent computing other path matrices
    """
    time_compute_pm2 = 0
    time_compute_pm = 0
    p = G.shape[0]

    # Compute path_matrix2 if not provided
    if path_matrix2 is None:
        path_matrix2 = compute_path_matrix2(G, cond_set, path_matrix)

    # Find ancestors of conditioning set
    if len(cond_set) == 0:
        anc_of_cond_set = []
    elif len(cond_set) == 1:
        anc_of_cond_set = np.where(path_matrix[:, cond_set[0]] > 0)[0].tolist()
    else:
        anc_of_cond_set = np.where(np.sum(path_matrix[:, cond_set], axis=1) > 0)[0].tolist()

    # Initialize the reachability matrix (2p x 2p)
    reachability_matrix = np.zeros((2 * p, 2 * p))
    reachable_on_non_causal_path_later = np.zeros((2, 2))

    # Initialize arrays to track reachable nodes
    reachable_nodes = np.zeros(2 * p)
    reachable_on_non_causal_path = np.zeros(2 * p)
    already_checked = np.zeros(p)

    # Initialize to_check with node i
    to_check = [0, 0]  # First two entries are placeholders

    # Find children of i (reachable with incoming edge)
    reachable_ch = np.where(G[i, :] == 1)[0].tolist()
    if len(reachable_ch) > 0:
        to_check.extend(reachable_ch)
        reachable_nodes[reachable_ch] = 1
        # Set these edges to 0 to avoid revisiting them
        G_temp = G.copy()
        G_temp[i, reachable_ch] = 0
    else:
        G_temp = G.copy()

    # Find parents of i (reachable with outgoing edge)
    reachable_pa = np.where(G[:, i] == 1)[0].tolist()
    if len(reachable_pa) > 0:
        to_check.extend(reachable_pa)
        # Mark parents as reachable with outgoing edge (index + p)
        reachable_nodes[np.array(reachable_pa) + p] = 1
        reachable_on_non_causal_path[np.array(reachable_pa) + p] = 1
        # Set these edges to 0
        G_temp[reachable_pa, i] = 0

    # Process nodes in to_check
    k = 2  # Start at index 2 (after placeholders)
    while k < len(to_check):
        a1 = to_check[k]
        k += 1

        if already_checked[a1] == 0:
            # Mark as checked
            current_node = a1
            already_checked[current_node] = 1

            # Process parents of current node
            pa = np.where(G_temp[:, current_node] == 1)[0].tolist()

            # Parents not in conditioning set
            pa1 = [p for p in pa if p not in cond_set]
            if len(pa1) > 0:
                reachability_matrix[pa1, current_node] = 1
                reachability_matrix[np.array(pa1) + p, current_node] = 1

            # If current node is in ancestors of conditioning set
            if current_node in anc_of_cond_set:
                if len(pa) > 0:
                    reachability_matrix[current_node, np.array(pa) + p] = 1
                    if path_matrix2[i, current_node] > 0:
                        new_rows = np.column_stack([np.repeat(current_node, len(pa)), pa])
                        reachable_on_non_causal_path_later = np.vstack([reachable_on_non_causal_path_later, new_rows])

                # Add new nodes to to_check
                new_to_check = [n for n in pa if already_checked[n] == 0]
                to_check.extend(new_to_check)

            # If current node is not in conditioning set
            if current_node not in cond_set:
                if len(pa) > 0:
                    reachability_matrix[current_node + p, np.array(pa) + p] = 1
                    new_to_check = [n for n in pa if already_checked[n] == 0]
                    to_check.extend(new_to_check)

            # Process children of current node
            ch = np.where(G_temp[current_node, :] == 1)[0].tolist()

            # Children not in conditioning set
            ch1 = [c for c in ch if c not in cond_set]
            if len(ch1) > 0:
                reachability_matrix[np.array(ch1) + p, current_node + p] = 1

            # Children that are ancestors of conditioning set
            ch2 = [c for c in ch if c in anc_of_cond_set]
            if len(ch2) > 0:
                reachability_matrix[ch2, current_node + p] = 1

                ch2b = [c for c in ch2 if path_matrix2[i, c] > 0]
                if len(ch2b) > 0:
                    new_rows = np.column_stack([ch2b, np.repeat(current_node, len(ch2b))])
                    reachable_on_non_causal_path_later = np.vstack([reachable_on_non_causal_path_later, new_rows])

            # If current node not in conditioning set, add children to to_check
            if current_node not in cond_set:
                if len(ch) > 0:
                    reachability_matrix[current_node, ch] = 1
                    reachability_matrix[current_node + p, ch] = 1
                    new_to_check = [n for n in ch if already_checked[n] == 0]
                    to_check.extend(new_to_check)

    # Compute full reachability
    reachability_path_matrix = compute_path_matrix(reachability_matrix, sparse)

    # Update reachable nodes using the reachability matrix
    ttt2 = np.where(reachable_nodes == 1)[0]
    if len(ttt2) == 1:
        tt2 = np.where(reachability_path_matrix[ttt2[0], :] > 0)[0]
    else:
        tt2 = np.where(np.sum(reachability_path_matrix[ttt2, :], axis=0) > 0)[0]

    reachable_nodes[tt2] = 1

    # First activation step
    ttt = np.where(reachable_on_non_causal_path == 1)[0]
    if len(ttt) == 1:
        tt = np.where(reachability_path_matrix[ttt[0], :] > 0)[0]
    else:
        tt = np.where(np.sum(reachability_path_matrix[ttt, :], axis=0) > 0)[0]

    reachable_on_non_causal_path[tt] = 1

    # Second activation step
    if reachable_on_non_causal_path_later.shape[0] > 2:
        for kk in range(2, reachable_on_non_causal_path_later.shape[0]):
            reachable_through = int(reachable_on_non_causal_path_later[kk, 0])
            new_reachable = int(reachable_on_non_causal_path_later[kk, 1])

            reachable_on_non_causal_path[new_reachable + p] = 1

            # Cancel connections
            reachability_path_matrix[new_reachable, reachable_through] = 0
            reachability_path_matrix[new_reachable, reachable_through + p] = 0
            reachability_path_matrix[new_reachable + p, reachable_through] = 0
            reachability_path_matrix[new_reachable + p, reachable_through + p] = 0

        # Update reachable_on_non_causal_path
        ttt = np.where(reachable_on_non_causal_path == 1)[0]
        if len(ttt) == 1:
            tt = np.where(reachability_path_matrix[ttt[0], :] > 0)[0]
        else:
            tt = np.where(np.sum(reachability_path_matrix[ttt, :], axis=0) > 0)[0]

        reachable_on_non_causal_path[tt] = 1

    # Combine results
    reachable_j = np.zeros(p, dtype=bool)
    for j in range(p):
        reachable_j[j] = reachable_nodes[j] > 0 or reachable_nodes[j + p] > 0

    reachable_on_non_causal_path_result = np.zeros(p, dtype=bool)
    for j in range(p):
        reachable_on_non_causal_path_result[j] = (
            reachable_on_non_causal_path[j] > 0 or reachable_on_non_causal_path[j + p] > 0
        )

    return reachable_on_non_causal_path_result, time_compute_pm2, time_compute_pm


def all_dags_intern(
    adj_mat: np.ndarray, a: np.ndarray, row_names: List[int], tmp: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Internal function for enumerating all DAGs consistent with an undirected component.

    Args:
        adj_mat: The adjacency matrix of the graph
        a: Submatrix of an undirected component
        row_names: Names/indices of the rows in the submatrix
        tmp: Temporary result matrix

    Returns:
        Matrix where each row represents a DAG
    """
    # Check if the matrix is entirely undirected
    if np.any((a + a.T) == 1):
        raise ValueError("The matrix is not entirely undirected. This should not happen!")

    # Base case: no edges in the submatrix
    if np.sum(a) == 0:
        # Reshape adjacency matrix into a vector (row-wise)
        adj_vec = adj_mat.flatten()

        # If tmp is None, initialize it
        if tmp is None:
            return np.array([adj_vec])

        # Add the new adjacency vector if it's not a duplicate
        tmp2 = np.vstack([tmp, adj_vec]) if tmp.size > 0 else np.array([adj_vec])

        # Check for duplicates
        if len(np.unique(tmp2, axis=0)) == len(tmp2):
            tmp = tmp2
    else:
        # Find potential sink nodes (those with neighbors)
        sinks = np.where(np.sum(a, axis=0) > 0)[0]

        for x in sinks:
            adj_mat2 = adj_mat.copy()

            # Check connectivity of neighbors
            adj = a == 1
            adj_x = adj[x, :]

            if np.any(adj_x):
                un = np.where(adj_x)[0]
                pp = len(un)
                adj2 = adj[np.ix_(un, un)]
                np.fill_diagonal(adj2, True)
            else:
                # x doesn't have any neighbors
                adj2 = np.array([[True]])

            # Check if all neighbors of x are connected
            if np.all(adj2):
                if np.any(adj_x):
                    un_indices = [row_names[i] for i in un]
                    row_x = row_names[x]

                    # Orient edges from neighbors to x
                    for u in un_indices:
                        adj_mat2[u, row_x] = 1
                        adj_mat2[row_x, u] = 0

                # Recursive call after removing node x
                a2 = np.delete(np.delete(a, x, axis=0), x, axis=1)
                row_names2 = row_names.copy()
                del row_names2[x]

                if tmp is None:
                    tmp = np.array([])

                tmp = all_dags_intern(adj_mat2, a2, row_names2, tmp)

    return tmp if tmp is not None else np.array([])


def all_dags_jonas(adj: np.ndarray, row_names: List[int]) -> np.ndarray:
    """
    Enumerate all DAGs consistent with a given CPDAG's undirected component.

    Args:
        adj: Adjacency matrix
        row_names: Names/indices of the rows in the submatrix

    Returns:
        Matrix where each row represents a DAG, or -1 if the input is invalid
    """
    # Extract the submatrix for the specified component
    a = adj[np.ix_(row_names, row_names)]

    # Check if the matrix is entirely undirected
    if np.any((a + a.T) == 1):
        return -1

    return all_dags_intern(adj, a, row_names, None)


def structural_intervention_distance(
    true_graph: np.ndarray, est_graph: np.ndarray, output: bool = False, sparse: bool = False
) -> dict:
    """
    Calculate the Structural Intervention Distance (SID) between two DAGs.

    Args:
        true_graph: Adjacency matrix of the true DAG (ground truth)
        est_graph: Adjacency matrix of the estimated DAG
        output: Whether to print detailed output
        sparse: Whether to use sparse matrix operations

    Returns:
        Dictionary containing:
        - sid: The SID value
        - sid_upper_bound: Upper bound of SID
        - sid_lower_bound: Lower bound of SID
        - incorrect_mat: Matrix showing incorrect interventions
    """
    # Convert input matrices to numpy arrays if they aren't already
    true_graph = np.array(true_graph)
    est_graph = np.array(est_graph)

    p = true_graph.shape[0]  # Number of nodes

    # Initialize matrices to track incorrect and correct interventions
    incorrect_int = np.zeros((p, p))
    correct_int = np.zeros((p, p))

    # Initialize counters
    minimum_total = 0
    maximum_total = 0
    time_path_matrix2 = 0
    time_all_compute_pm2 = 0
    time_all_compute_pm = 0
    time_all_dsep = 0
    time_exp_graph = 0
    num_checks = 0

    # Compute the path matrix
    path_matrix = compute_path_matrix(true_graph, sparse)

    # Compute undirected components of estimated graph
    est_undir = est_graph * est_graph.T

    try:
        import networkx as nx

        # Convert to NetworkX graph to find connected components
        g = nx.from_numpy_array(est_undir, create_using=nx.Graph())
        conn_comp = list(nx.connected_components(g))
    except ImportError:
        # Fallback if NetworkX is not available - simple implementation
        # This is a simplified version and may not be as efficient
        # as the NetworkX implementation
        def find_components(adj_matrix):
            n = adj_matrix.shape[0]
            visited = [False] * n
            components = []

            for i in range(n):
                if not visited[i]:
                    component = []
                    queue = [i]
                    visited[i] = True

                    while queue:
                        node = queue.pop(0)
                        component.append(node)

                        for j in range(n):
                            if adj_matrix[node, j] > 0 and not visited[j]:
                                visited[j] = True
                                queue.append(j)

                    components.append(component)

            return components

        conn_comp = find_components(est_undir)

    num_conn_comp = len(conn_comp)
    est_graph_is_essential = True

    # Check each connected component
    for ll in range(num_conn_comp):
        comp = list(conn_comp[ll])

        if len(comp) > 1:
            try:
                import networkx as nx

                # Check if the component is chordal
                subgraph = nx.from_numpy_array(est_undir[np.ix_(comp, comp)], create_using=nx.Graph())
                is_chordal = nx.is_chordal(subgraph)

                if not is_chordal:
                    if output:
                        print("The estimated graph is not chordal. We consider local expansions.")
                    est_graph_is_essential = False

                if len(comp) > 8:
                    if output:
                        print("The connected component is too large (>8 nodes). Using local expansions.")
                    est_graph_is_essential = False
            except ImportError:
                # Simplified check - assume not chordal
                if output:
                    print("NetworkX not available, assuming graph is not chordal")
                est_graph_is_essential = False

    # Process each connected component
    for ll in range(num_conn_comp):
        comp = list(conn_comp[ll])

        if len(comp) > 0:
            if est_graph_is_essential:
                # Expand the connected component into DAGs
                if len(comp) > 1:
                    mmm = all_dags_jonas(est_graph, comp)
                else:
                    # For a single node, there's only one possible graph
                    mmm = np.array([est_graph.flatten()])

                if isinstance(mmm, int) and mmm == -1:
                    est_graph_is_essential = False
                    mmm = np.array([est_graph.flatten()])

                # Prepare for tracking incorrect interventions
                if mmm.size > 0:
                    dim_m = mmm.shape
                    incorrect_sum = np.zeros(dim_m[0])
                else:
                    est_graph_is_essential = False
                    if output:
                        print("Something is wrong. The estimated graph might not be a CPDAG.")

            # If not an essential graph or expansion failed, use local approach
            if not est_graph_is_essential:
                incorrect_sum = np.zeros(1)  # Initialize for the single local expansion
                mmm = np.array([est_graph.flatten()])  # Use the original graph

        # Process each node in the component
        for i in comp:
            # Get parents in true and estimated graphs
            pa_true = np.where(true_graph[:, i] == 1)[0].tolist()

            # Certain parents in estimated graph (directed edges)
            certain_pa_est = np.where((est_graph[:, i] == 1) & (est_graph[i, :] == 0))[0].tolist()

            # Possible parents (undirected edges)
            possible_pa_est = np.where((est_graph[:, i] == 1) & (est_graph[i, :] == 1))[0].tolist()

            if not est_graph_is_essential:
                # Consider all local combinations of parents
                max_count = 2 ** len(possible_pa_est)
                unique_rows = list(range(max_count))

                # Create all possible combinations for parent sets
                mmm = np.tile(est_graph.flatten(), (max_count, 1))

                # Generate all possible parent assignments using binary combinations
                if len(possible_pa_est) > 0:
                    import itertools

                    parent_combinations = list(itertools.product([0, 1], repeat=len(possible_pa_est)))

                    for idx, comb in enumerate(parent_combinations):
                        for j, val in zip(possible_pa_est, comb):
                            # Set edges according to the combination
                            mmm[idx, i + j * p] = val

                incorrect_sum = np.zeros(max_count)
            else:
                if mmm.shape[0] > 1:
                    # Find unique parent sets for node i across all DAG expansions
                    all_parents_of_i = list(range(i, p * p, p))

                    # Find unique rows based on parent patterns
                    unique_parent_patterns = {}
                    for row_idx in range(mmm.shape[0]):
                        pattern = tuple(mmm[row_idx, all_parents_of_i])
                        if pattern not in unique_parent_patterns:
                            unique_parent_patterns[pattern] = row_idx

                    unique_rows = list(unique_parent_patterns.values())
                    max_count = len(unique_rows)
                else:
                    max_count = 1
                    unique_rows = [0]

            # Process each possible parent set
            count = 1
            while count <= max_count:
                if max_count == 1:
                    pa_est = certain_pa_est
                else:
                    # Extract the graph for this expansion
                    est_graph_new = mmm[unique_rows[count - 1], :].reshape(p, p).T
                    pa_est = np.where(est_graph_new[:, i] == 1)[0].tolist()

                    if output:
                        print(f"Node {i} has {len(pa_est)} parents in expansion {unique_rows[count-1]}:")
                        print(pa_est)

                # Compute path matrix with edges from pa_est removed
                path_matrix2 = compute_path_matrix2(true_graph, pa_est, path_matrix, sparse)

                # Check d-separations
                reachable_wo_causal_path, _, _ = find_reachable_on_non_directed_path(
                    true_graph, i, pa_est, path_matrix, path_matrix2, sparse
                )
                num_checks += 1

                # Check each target node for intervention correctness
                for j in range(p):
                    if i != j:  # Skip self-interventions
                        finished = False
                        ij_true_null = False
                        ij_est_null = False

                        # Check if intervention effect is zero in true graph
                        if path_matrix[i, j] == 0:
                            ij_true_null = True

                        # Check if target is a parent in estimated graph
                        if j in pa_est:
                            ij_est_null = True

                        # If both agree on zero effect
                        if ij_est_null and ij_true_null:
                            finished = True
                            correct_int[i, j] = 1

                        # If estimated predicts zero but true doesn't
                        if ij_est_null and not ij_true_null:
                            incorrect_int[i, j] = 1
                            incorrect_sum[unique_rows[count - 1]] += 1

                            # Add to all entries with the same parent set
                            if est_graph_is_essential and mmm.shape[0] > 1:
                                parent_pattern = tuple(mmm[unique_rows[count - 1], all_parents_of_i])
                                for row_idx in range(mmm.shape[0]):
                                    if (
                                        row_idx not in unique_rows
                                        and tuple(mmm[row_idx, all_parents_of_i]) == parent_pattern
                                    ):
                                        incorrect_sum[row_idx] += 1

                            finished = True

                        # If parent sets are identical
                        if not finished and set(pa_true) == set(pa_est):
                            finished = True
                            correct_int[i, j] = 1

                        # More complex checks for intervention effects
                        if not finished:
                            if path_matrix[i, j] > 0:
                                # Find children on causal paths to j
                                chi_caus_path = []
                                for child in np.where(true_graph[i, :] == 1)[0]:
                                    if path_matrix[child, j] > 0:
                                        chi_caus_path.append(child)

                                # Check if any descendant of a child on causal path is in pa_est
                                descendants_in_pa_est = False
                                for child in chi_caus_path:
                                    if any(path_matrix[child, pa] > 0 for pa in pa_est):
                                        descendants_in_pa_est = True
                                        break

                                if descendants_in_pa_est:
                                    incorrect_int[i, j] = 1
                                    incorrect_sum[unique_rows[count - 1]] += 1

                                    # Add to all entries with the same parent set
                                    if est_graph_is_essential and mmm.shape[0] > 1:
                                        parent_pattern = tuple(mmm[unique_rows[count - 1], all_parents_of_i])
                                        for row_idx in range(mmm.shape[0]):
                                            if (
                                                row_idx not in unique_rows
                                                and tuple(mmm[row_idx, all_parents_of_i]) == parent_pattern
                                            ):
                                                incorrect_sum[row_idx] += 1

                                    finished = True

                            if not finished:
                                # Check if there are non-causal paths not blocked by conditioning set
                                if reachable_wo_causal_path[j]:
                                    incorrect_int[i, j] = 1
                                    incorrect_sum[unique_rows[count - 1]] += 1

                                    # Add to all entries with the same parent set
                                    if est_graph_is_essential and mmm.shape[0] > 1:
                                        parent_pattern = tuple(mmm[unique_rows[count - 1], all_parents_of_i])
                                        for row_idx in range(mmm.shape[0]):
                                            if (
                                                row_idx not in unique_rows
                                                and tuple(mmm[row_idx, all_parents_of_i]) == parent_pattern
                                            ):
                                                incorrect_sum[row_idx] += 1
                                else:
                                    correct_int[i, j] = 1

                count += 1

            # Update minimum/maximum totals
            if not est_graph_is_essential:
                minimum_total += min(incorrect_sum)
                maximum_total += max(incorrect_sum)
                incorrect_sum = np.zeros(1)

            # Print details if requested
            if len(incorrect_sum) > 1 and output:
                print(f"For variable {i}, we have more than one possible set of parents.")
                print("Vector of incorrect interventions for different parent sets:")
                print(incorrect_sum)

        # Update minimum/maximum after processing the component
        minimum_total += min(incorrect_sum) if len(incorrect_sum) > 0 else 0
        maximum_total += max(incorrect_sum) if len(incorrect_sum) > 0 else 0
        incorrect_sum = np.zeros(1)

    # Print results if requested
    if output and p < 11:
        print("Incorrectly predicted interventions:")
        print(incorrect_int)
        print("Correctly predicted interventions:")
        print(correct_int)

    # Return results
    result = {
        "sid": int(np.sum(incorrect_int)),
        "sid_upper_bound": int(maximum_total),
        "sid_lower_bound": int(minimum_total),
        "incorrect_mat": incorrect_int,
    }

    if output:
        print(f"SID: {result['sid']}")
        print(f"SID upper bound: {result['sid_upper_bound']}")
        print(f"SID lower bound: {result['sid_lower_bound']}")
        print(f"Number of d-sep checks: {num_checks}")

    return result


def test_sid():
    """Run a simple test of the SID calculation."""
    # Example from the paper - Figure 2
    # True graph G
    G = np.zeros((5, 5))
    G[0, 1] = 1  # X1 -> X2
    G[0, 2] = 1  # X1 -> Y2
    G[1, 2] = 1  # X2 -> Y2
    G[0, 3] = 1  # X1 -> Y1
    G[1, 3] = 1  # X2 -> Y1
    G[0, 4] = 1  # X1 -> Y3
    G[1, 4] = 1  # X2 -> Y3
    G[2, 4] = 1  # Y2 -> Y3

    # Estimated graph H1 with extra edge Y1 -> Y2
    H1 = G.copy()
    H1[3, 2] = 1  # Y1 -> Y2

    # Estimated graph H2 with reversed edge X1 <-> X2
    H2 = G.copy()
    H2[0, 1] = 0  # Remove X1 -> X2
    H2[1, 0] = 1  # Add X2 -> X1

    # Calculate SID
    result_H1 = structural_intervention_distance(G, H1, output=False)
    assert result_H1["sid"] == 0, "Incorrect SID for H1"

    result_H2 = structural_intervention_distance(G, H2, output=False)
    assert result_H2["sid"] == 8, "Incorrect SID for H2"


if __name__ == "__main__":
    test_sid()
