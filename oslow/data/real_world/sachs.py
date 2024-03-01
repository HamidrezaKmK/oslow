import networkx as nx

from oslow.data import OCDDataset
import pandas as pd
import os
import typing as th

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


class SachsOCDDataset(OCDDataset):
    def __init__(self, standard: bool = False):
        # load csv file into pandas dataframe
        df = pd.read_csv(os.path.join(_DATA_DIR, "sachs/sachs.csv"))
        label_mapping = {
            0: "Raf",
            1: "Mek",
            2: "Plcg",
            3: "PIP2",
            4: "PIP3",
            5: "Erk",
            6: "Akt",
            7: "PKA",
            8: "PKC",
            9: "P38",
            10: "Jnk",
        }
        inverse_mapping = {v: k for k, v in label_mapping.items()}
        df.rename(columns=inverse_mapping, inplace=True)

        graph = {
            2: [3, 4],
            4: [3],
            7: [1, 5, 6, 10, 9, 0],
            8: [1, 7, 10, 0, 9],
            0: [1],
            1: [5],
            5: [6],
        }
        graph = nx.DiGraph(graph)

        super().__init__(samples=df, dag=graph, name="sachs", standard=standard)
