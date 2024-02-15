from typing import Optional, Literal, List
import numpy as np
import networkx as nx
import random


class GraphGenerator:
    """
    A graph generator class that creates networkx graphs
    using a type and appropriate arguments.

    Using the generate_dag method, a random graph can be generated and returned.
    By calling the generate_dag method multiple times, different random graphs can be generated and using a seed, the same graph can be generated again.
    """

    def __init__(
        self,
        num_nodes: int,
        seed: Optional[int] = None,
        base_dag: Optional[np.array] = None,
        graph_type: Optional[
            Literal[
                "erdos_renyi",
                "barabasi_albert",
                "random_dag",
                "chain",
                "collider",
                "full",
                "tree",
                "fork",
                "v_structure",
            ]
        ] = None,
        enforce_ordering: Optional[List] = None,
        **graph_generation_args,
    ) -> None:
        self.num_nodes = num_nodes
        self.seed = seed
        self.base_dag = base_dag
        self.graph_generator_type = graph_type
        self.graph_generator_args = graph_generation_args
        self.enforce_ordering = enforce_ordering

    def generate_dag(self, seed: Optional[int] = None) -> nx.DiGraph:
        """
        If base_dag is given, then create a dag based on that.
        Otherwise, generate a random graph using the specified graph generator
        and then make it Acyclic via a random permutation.

        Returns:
            nx.Graph: a random graph
        """
        if seed is None:
            seed = self.seed
            self.seed += 1

        # create a nx.Graph using base_dag as the adjacency matrix
        if self.base_dag is not None:
            new_dag = nx.DiGraph(self.base_dag)
            return new_dag

        # if the base graph is not given, generate the graph using a generator
        if self.graph_generator_type == "erdos_renyi":
            # generate a random graph using erdos renyi model
            if "p" not in self.graph_generator_args:
                raise ValueError(
                    "p must be specified when using erdos_renyi graph generator"
                )
            new_dag = nx.gnp_random_graph(
                self.num_nodes, self.graph_generator_args["p"], seed=seed
            )
        elif self.graph_generator_type == "barabasi_albert":
            # generate a random graph using barabasi albert model
            if "m" not in self.graph_generator_args:
                raise ValueError(
                    "m must be specified when using barabasi_albert graph generator"
                )
            new_dag = nx.barabasi_albert_graph(
                self.num_nodes, self.graph_generator_args["m"], seed=seed
            )
        elif self.graph_generator_type == "random_dag":
            # generate a random graph
            if "m" not in self.graph_generator_args:
                raise ValueError(
                    "m must be specified when using random_dag graph generator"
                )
            new_dag = nx.gnm_random_graph(
                self.num_nodes, self.graph_generator_args["m"], seed=seed
            )
        elif self.graph_generator_type == "chain":
            # generate a chain graph
            new_dag = nx.path_graph(self.num_nodes)
        elif self.graph_generator_type in ["collider", "fork", "v_structure"]:
            # generate a star graph
            new_dag = nx.star_graph(self.num_nodes - 1)
            # swap the first and last node
            if self.graph_generator_type in ["collider", "v_structure"]:
                all_nodes = list(new_dag.nodes.keys())
                new_dag = nx.relabel_nodes(
                    new_dag, {all_nodes[0]: all_nodes[-1], all_nodes[-1]: all_nodes[0]}
                )
        elif self.graph_generator_type == "full":
            # generate a complete graph
            new_dag = nx.complete_graph(self.num_nodes)
        elif self.graph_generator_type == "tree":
            # generate a jungle graph
            new_dag = nx.random_tree(self.num_nodes, seed=seed)

        else:
            raise ValueError(
                f"dag_generator_type {self.graph_generator_type} not found!\nIf not then base_dag must be specified when base_dag is not provided"
            )
        # make new_dag directed acyclic graph

        # create a directed graph from new_dag
        new_dag = nx.DiGraph(new_dag)

        # iterate over all the edges of new_dag
        delete_edges = []
        for edge in new_dag.edges:
            # check if the edge is directed from a node with a higher index to a node with a lower index
            # get the location of edge[0] and edge[1] in the permutation
            u = (
                edge[0]
                if isinstance(edge[0], int)
                else list(new_dag.nodes).index(edge[0])
            )
            v = (
                edge[1]
                if isinstance(edge[1], int)
                else list(new_dag.nodes).index(edge[1])
            )
            if u >= v:
                # remove every such edge
                delete_edges.append((edge[0], edge[1]))

        new_dag.remove_edges_from(delete_edges)

        # permute the nodes of new_dag if the ordering is not enforced, otherwise, do it according to the ordering
        correct_ordering = nx.topological_sort(new_dag)

        np.random.seed(seed=seed)
        random.seed(seed)
        # if self.enforce_ordering is None then create a random ordering
        if self.enforce_ordering is None:
            ordering = list(range(self.num_nodes))
            random.shuffle(ordering)
        else:
            ordering = self.enforce_ordering

        new_dag = nx.relabel_nodes(new_dag, dict(zip(correct_ordering, ordering)))
        return new_dag
