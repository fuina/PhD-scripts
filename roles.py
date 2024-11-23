from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
from cdlib import algorithms


def get_role(z, P):
    if z < 2.5:
        if P == 0:
            return 'R1'
        elif P < 0.625:
            return 'R2'
        elif P < 0.8:
            return 'R3'
        else:
            return 'R4'
    else:
        if P < 0.3:
            return 'R5'
        elif P < 0.75:
            return 'R6'
        else:
            return 'R7'


def get_communities(graph):
    if len(nx.get_node_attributes(graph, 'community')) == 0:
        print(datetime.now().time(), 'Computing communities...')
        communities = algorithms.louvain(graph.to_undirected())
        graph.graph['modularity'] = communities.newman_girvan_modularity()[2]
        communities = communities.to_node_community_map()
        communities = {k: v[0] for k, v in communities.items()}
        nx.set_node_attributes(graph, communities, 'community')
    return nx.get_node_attributes(graph, 'community')


def get_within_community_degrees(graph,weight='weight'):
    if len(nx.get_node_attributes(graph, 'z')) == 0:
        print(datetime.now().time(), 'Computing within community degrees...')
        s = get_communities(graph)
        def get_subgraph_nodes(x): return {i for i in graph.nodes() if s[i] == x}
        def get_subgraph(x): return graph.subgraph(get_subgraph_nodes(x))
        k = {
            si: get_subgraph(si).in_degree(weight=weight)
            for si in set(s.values())
        }
        mean_k = {
            si: np.mean(list(dict(k[si]).values()))
            for si in set(s.values())
        }
        std_k = {
            si: np.std(list(dict(k[si]).values()))
            for si in set(s.values())
        }
        z = {i: (k[s[i]][i] - mean_k[s[i]])/std_k[s[i]] for i in graph.nodes()}
        nx.set_node_attributes(graph, z, 'z')
    return nx.get_node_attributes(graph, 'z')


def get_participation_coefficients(graph,weight='weight'):
    if len(nx.get_node_attributes(graph, 'P')) == 0:
        print(datetime.now().time(), 'Computing participation coefficients...')
        s = get_communities(graph)
        P = dict()
        for i in graph.nodes():
            k = [(s[j], w) for j, _, w in graph.in_edges(i, data=weight)]
            k += [(s[j], w) for _, j, w in graph.out_edges(i, data=weight)]
            if len(k) == 0:
                continue
            k = pd.DataFrame(k).groupby(by=0).sum()
            k[1] = k[1] * k[1] / sum(k[1]) / sum(k[1])
            P[i] = 1 - sum(k[1])
        nx.set_node_attributes(graph, P, 'P')
    return nx.get_node_attributes(graph, 'P')


def get_roles(graph,weight='weight'):
    if len(nx.get_node_attributes(graph, 'role')) == 0:
        print(datetime.now().time(), 'Computing roles...')
        z = get_within_community_degrees(graph,weight)
        P = get_participation_coefficients(graph,weight)
        roles = {n: get_role(z[n], P[n]) for n in graph.nodes() if n in P}
        nx.set_node_attributes(graph, roles, 'role')
    return nx.get_node_attributes(graph, 'role')

