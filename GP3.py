import networkx as nx
import numpy as np
from utils.cutom_env import *
from scipy.stats import norm


def weight_func(k):
    return lambda u, v, d: min(k * attr.get("mu", 1) + attr.get("sigma2", 1) for attr in d.values())


def gp3_query(mymap, k, OD, T):
    p = nx.dijkstra_path(G=mymap.G, source=OD[0]-1, target=OD[1]-1, weight='mu')
    cur_mu = np.sum(mymap.mu[p])
    cur_sigma = np.sqrt(np.sum(mymap.sigma2[p]))
    best_prob = norm(cur_mu, cur_sigma).cdf(T)
    for i in range(k):
        q = nx.dijkstra_path(mymap.G, source=OD[0]-1, target=OD[1]-1, weight=weight_func(k=i))
        cur_mu = np.sum(mymap.mu[q])
        cur_sigma = np.sqrt(np.sum(mymap.sigma2[q]))
        cur_prob = norm(cur_mu, cur_sigma).cdf(T)
        if cur_prob > best_prob:
            best_prob = cur_prob
            p = q
    return np.array(p) + 1, best_prob


if __name__ == '__main__':
    map1 = MapInfo("maps/sioux_network.csv")
    path, prob = gp3_query(map1, 100, [1, 15], 43)
    map1.get_sample_time(path)


