import os, sys
import numpy as np
import dypy
from pathlib import Path
import yaml
import pandas as pd
from pprint import pprint

sys.path.append("..")

_DIR = os.path.dirname(os.path.abspath(__file__))
_REAL_WORLD_DIR = os.path.join(_DIR, "experiments", "real-world")
PRUNE_FILE = os.path.join(_DIR, "experiments", "results", "prune_results.csv")

# SYNTREN_FILE = os.path.join(_DIR, "experiments", "results", "syntren-ours.csv")

from oslow.evaluation import count_SHD, count_SID, count_backward
from oslow.post_processing.cam_pruning import sparse_regression_based_pruning
from oslow.post_processing.pc_pruning import pc_based_pruning
from oslow.post_processing.ultimate_pruning import ultimate_pruning

import typing as th
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import kci
import json

import argparse

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='cam', type=str, choices=['cam', 'pc', 'ultimate'])
    parser.add_argument("--data_type", default="syntren", type=str, choices=['syntren', 'sachs'])
    parser.add_argument("--data_num", default=0, type=int)
    parser.add_argument("--order", default=None, type=str, required=False)
    parser.add_argument("--saved_permutations_dir", default=0, type=str, required=False)
    args = parser.parse_args()
    if args.order is None and args.saved_permutations_dir is None:
        raise ValueError("Either order or saved_permutations_dir must be provided")
    if args.order is None:
        # load the json from the saved_permutations_dir
        with open(os.path.join(args.saved_permutations_dir, "final-results.json"), "r") as f:
            permutations = json.load(f)
            args.order = permutations['most_common_permutation']
    return args


def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d,d))
    for i, var in enumerate(top_order):
        A[var, top_order[i+1:]] = 1
    return A

def get_data_config(data_type, data_num):
    if data_type == "syntren":
        paths = Path(_REAL_WORLD_DIR).glob("data-syntren-*.yaml")
    else:
        paths = Path(_REAL_WORLD_DIR).glob("data-sachs.yaml")
    paths = sorted(list(paths))
    data_path = paths[data_num]
    data_config = yaml.load(open(data_path, "r"), Loader=yaml.FullLoader)["init_args"]
    data_name = str(data_path).split("/")[-1].split(".")[0]
    return data_config


def metrics(order: str, data_config, method="pc"):
    order = [int(n) for n in order.split('-')]
    dag = full_DAG(order)
    dataset = dypy.get_value(data_config["dataset"])
    dataset_args = data_config["dataset_args"] if "dataset_args" in data_config else {}
    dataset = dataset(**dataset_args)
    true_dag = dataset.dag
    X = dataset.samples
    if method == "pc":
        dag = pc_based_pruning(X, order, verbose=False)
    elif method == "cam":
        dag = sparse_regression_based_pruning(X, order)
    elif method == "ultimate":
        dag = ultimate_pruning(X, order)
    else:
        raise NotImplementedError()
    return {'SID': count_SID(true_dag, dag), 'SHD': count_SHD(true_dag, dag)}


# config = get_data_config("sachs", 0)

# print(metrics("7-0-8-2-4-1-9-10-3-5-6", config))

def save_results(results_dict):
    # get the directory name from PRUNE_FILE
    dirname = os.path.dirname(PRUNE_FILE)
    if not os.path.exists(PRUNE_FILE):
        # create the directory if it does not exist
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        pd.DataFrame([results_dict]).to_csv(PRUNE_FILE, index=False)
    else:
        df = pd.read_csv(PRUNE_FILE)
        df.loc[len(df.index)] = results_dict
        df.to_csv(PRUNE_FILE, index=False)
    
    print("Results: ")
    pprint(results_dict)
    print("Results saved to ", PRUNE_FILE)

if __name__ == "__main__":
    the_args = build_args()
    
    print("----")
    print("Running pruning with:")
    print("method:", the_args.method)
    print("data_type:", the_args.data_type)
    print("data_num:", the_args.data_num)
    print("----")
    
    if the_args.data_type == "syntren":
        data_config = get_data_config("syntren", the_args.data_num)
        # df = pd.read_csv(SYNTREN_FILE)
        # order = df[df['1dataset'] == the_args.data_num]['permutation'].item()
    elif the_args.data_type == "sachs":
        data_config = get_data_config("sachs", the_args.data_num)
    else:
        raise NotImplementedError(f"{the_args.data_type} not implemented")
    
    if the_args.order is not None:
        order = the_args.order
    else:
        raise ValueError("order not available")
    
    
    res = metrics(order, data_config, method=the_args.method)
    res['name'] = the_args.data_type
    res['num'] = the_args.data_num
    res['method'] = the_args.method
    save_results(res)