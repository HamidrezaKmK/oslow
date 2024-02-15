# Codes are adopted from the original implementation
# https://github.com/paulrolland1307/SCORE/tree/5e18c73a467428d51486d2f683349dde2607bfe1
# under the GNU Affero General Public License v3.0
# Copy right belongs to the original author https://github.com/paulrolland1307

import os
import uuid

from cdt.utils.R import launch_R_script
import pandas as pd
import numpy as np
import typing as th
import networkx as nx

_DIR = os.path.dirname(os.path.abspath(__file__))


def np_to_csv(array, save_path):
    """
    Convert np array to .csv
    array: numpy array
        the numpy array to convert to csv
    save_path: str
        where to temporarily save the csv
    Return the path to the csv file
    """
    id = str(uuid.uuid4())
    output = os.path.join(os.path.dirname(save_path), "tmp_" + id + ".csv")

    df = pd.DataFrame(array)
    df.to_csv(output, header=False, index=False)

    return output


def cam_pruning(adj_matrix: np.ndarray, data: np.ndarray, cutoff: float = 0.001, verbose: bool = False):
    save_path = os.path.join(_DIR, "score_cam_pruning")

    data_csv_path = np_to_csv(data, save_path)
    dag_csv_path = np_to_csv(adj_matrix, save_path)

    arguments = dict()
    arguments["{PATH_DATA}"] = data_csv_path
    arguments["{PATH_DAG}"] = dag_csv_path
    arguments["{PATH_RESULTS}"] = os.path.join(save_path, "results.csv")
    arguments["{ADJFULL_RESULTS}"] = os.path.join(save_path, "adjfull.csv")
    arguments["{CUTOFF}"] = str(cutoff)
    arguments["{VERBOSE}"] = "TRUE" if verbose else "FALSE"

    def retrieve_result():
        A = pd.read_csv(arguments["{PATH_RESULTS}"]).values
        os.remove(arguments["{PATH_RESULTS}"])
        os.remove(arguments["{PATH_DATA}"])
        os.remove(arguments["{PATH_DAG}"])
        return A

    dag = launch_R_script(
        os.path.join(_DIR, "score_cam_pruning/cam_pruning.R"), arguments, output_function=retrieve_result
    )
    return dag


def sparse_regression_based_pruning(df: pd.DataFrame, ordering: th.List, verbose=False, cutoff: float = 0.001):
    df_np = df.to_numpy()
    full_graph = np.zeros((df_np.shape[1], df_np.shape[1]))
    for i, v in enumerate(ordering):
        for j, u in enumerate(ordering):
            if i >= j:
                continue
            # find the index of u and v in the dataframe columns
            u_idx = df.columns.get_loc(u)
            v_idx = df.columns.get_loc(v)
            full_graph[v_idx, u_idx] = 1

    if not verbose:
        import sys

        out = sys.stdout
        sys.stdout = open(os.devnull, "w")
    dag = cam_pruning(full_graph, df_np, cutoff=cutoff, verbose=verbose)
    if not verbose:
        sys.stdout = out

    all_edges = []
    for i in range(df_np.shape[1]):
        for j in range(df_np.shape[1]):
            if dag[i, j] == 1:
                all_edges.append((df.columns[i], df.columns[j]))

    g = nx.DiGraph()
    g.add_nodes_from(ordering)
    g.add_edges_from(all_edges)
    return g
