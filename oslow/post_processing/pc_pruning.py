"""
Here lies the code that gets the ordering and outputs the structure itself.
The entire process of causal discovery is split into order discovery and then 
pruning. This is where the pruning is done.
"""
import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import networkx as nx
import typing as th


from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import kci, gsq

indep_method = {
    'kci': kci,
    'g2': gsq
}

def pc_based_pruning(df: pd.DataFrame, ordering: th.List, independence_test: th.Literal['kci', 'g2']='kci', verbose=True, alpha=0.05):
    # change the dataframe to a numpy array
    df_np = df.to_numpy()
    global indep_method
    if independence_test not in indep_method:
        raise Exception(f"Independence test {independence_test} is not supported")
    cg = pc(df_np, alpha, indep_method[independence_test], verbose=verbose)
    graph = cg.G.graph
    list_all_edges = []
    for i, v in enumerate(ordering):
        for j, u in enumerate(ordering):
            if i >= j:
                continue
            u_idx = df.columns.get_loc(u)
            v_idx = df.columns.get_loc(v)
            if graph[u_idx, v_idx] != 0 or graph[v_idx, u_idx] != 0:
                list_all_edges.append((v, u))
    
    g = nx.DiGraph()
    g.add_nodes_from(ordering)
    g.add_edges_from(list_all_edges)
    return g
