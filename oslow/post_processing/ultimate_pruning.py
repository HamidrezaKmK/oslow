import pandas as pd
import typing as th
from .pc_pruning import pc_based_pruning
from .cam_pruning import cam_pruning
import networkx as nx
import numpy as np
import os

def ultimate_pruning(df: pd.DataFrame, ordering: th.List, independence_test: th.Literal['kci', 'g2']='kci', verbose=True, alpha=0.05, cutoff=0.001):
    # change the dataframe to a numpy array
    pruned_graph = pc_based_pruning(df, ordering, independence_test, verbose, alpha)
    
    # get the skeleton of pruned_graph
    skeleton = nx.Graph(pruned_graph)
    
    
    if verbose:
        print("Pruned graph after PC-based pruning and this is the skeleton:")
        for edge in skeleton.edges():
            print(edge)
        print("----")
        
    # find all the connected components of the skeleton
    connected_components = list(nx.connected_components(skeleton))
    
    if verbose:
        print("The connected components of the skeleton are:")
        for component in connected_components:
            print(component)
        print("----")
    
    df_np = df.to_numpy()
    full_graph = np.zeros((df_np.shape[1], df_np.shape[1]))
    for i, v in enumerate(ordering):
        for j, u in enumerate(ordering):
            if i >= j:
                continue
            # find the index of u and v in the dataframe columns
            u_idx = df.columns.get_loc(u)
            v_idx = df.columns.get_loc(v)
            # check if u and v are in the same connected component
            u_component = None
            v_component = None
            for component in connected_components:
                if u in component:
                    u_component = component
                if v in component:
                    v_component = component
            if u_component == v_component:
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
    