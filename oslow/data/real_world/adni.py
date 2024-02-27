import networkx as nx

from oslow.data import OCDDataset
import pandas as pd
from numpy.random import RandomState
import os

from sklearn.preprocessing import LabelEncoder

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


class ADNIOCDDataset(OCDDataset):
    def __init__(self, standard: bool = False):
        # load csv file into pandas dataframe
        df = self._load_preprocess()
        label_mapping = {
            0: "AGE",
            1: "PTGENDER",
            2: "PTEDUCAT",
            3: "FDG",
            4: "ABETA",
            5: "PTAU",
            6: "APOE4",
            7: "DX",
        }
        inverse_mapping = {v: k for k, v in label_mapping.items()}
        df.rename(columns=inverse_mapping, inplace=True)

        graph = {0: [4], 1: [], 2: [7], 3: [7], 4: [3, 5], 5: [7], 6: [4], 7: []}
        graph = nx.DiGraph(graph)

        super().__init__(samples=df, dag=graph, name="ADNI", standard=standard)

    def _load_preprocess(self):
        data = pd.read_csv(
            os.path.join(_DATA_DIR, "adni/ADNIMERGE.csv"), low_memory=False
        )
        studies = ["ADNI1", "ADNI2", "ADNIGO"]
        columns = [
            "RID",
            "EXAMDATE",
            "AGE",
            "PTGENDER",
            "PTEDUCAT",
            "FDG",
            "ABETA",
            "PTAU",
            "APOE4",
            "DX",
        ]
        data = (
            data.query("COLPROT in @studies")[columns]
            .dropna()
            .drop(["RID", "EXAMDATE"], axis=1)
        )

        data = data.replace({"ABETA": {">1700": 1700}, "PTAU": {">120": 120, "<8": 8}})

        discrete_cols = ["DX", "PTGENDER", "APOE4", "PTEDUCAT"]
        le = LabelEncoder()

        # Dequantize discrete columns
        prng = RandomState(42)
        for c in discrete_cols:
            data[c] = le.fit_transform(data[c]) + prng.uniform(
                0, 1, size=data[c].shape[0]
            )

        data["ABETA"] = data["ABETA"].astype("float64")
        data["PTAU"] = data["PTAU"].astype("float64")

        return data
