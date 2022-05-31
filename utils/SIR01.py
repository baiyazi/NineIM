import networkx as nx
import random
import matplotlib.pyplot as plt


def sir_one(graph, seeds, beta=0.4, mu=0.1):
    """单接触SIR模型
    传染率beta；恢复率mu"""
    I_set = set(seeds)
    S_set = set(graph.nodes()).difference(I_set)
    R_set = set()

    t = 0

    while len(I_set) > 0:
        Ibase = I_set.copy()
        for i in Ibase:
            # I --> R
            if random.random() < mu:
                I_set.remove(i)
                R_set.add(i)

            # S --> I
            if random.random() < beta:
                tmp = list(set(graph.neighbors(i)).intersection(S_set).copy())
                if len(tmp) > 0:
                    s = random.choice(tmp)
                    S_set.remove(s)
                    I_set.add(s)

        t += 1

    per_R = len(R_set) / len(graph.nodes())
    # print(t)
    # print("The percentage of R: " + str(per_R))

    return len(R_set)


def sir_more(graph, seeds, beta=0.4, mu=1):
    """全接触SIR模型
    传染率beta；恢复率mu"""
    I_set = set(seeds)
    S_set = set(graph.nodes()).difference(I_set)
    R_set = set()

    t = 0

    while len(I_set) > 0:
        Ibase = I_set.copy()
        for i in Ibase:
            # I --> R
            if random.random() < mu:
                I_set.remove(i)
                R_set.add(i)

            # S --> I
            tmp = list(set(graph.neighbors(i)).intersection(S_set).copy())
            for s in tmp:
                if random.random() < beta:
                    S_set.remove(s)
                    I_set.add(s)

        t += 1

    per_R = len(R_set) / len(graph.nodes())
    # print(t)
    # print("The percentage of R: " + str(per_R))

    return len(R_set)


def centrality_index(graph, num):
    """复杂网络中心性指标"""
    # degree
    degree_graph = dict(nx.degree(graph))
    trans_degree = list(zip(degree_graph.values(), degree_graph.keys()))
    trans_degree.sort(reverse=True)
    print(trans_degree)
    degree_seed = []
    for i in range(num):
        degree_seed.append(trans_degree[i][1])

    # core
    core_graph = dict(nx.core_number(graph))
    trans_core = list(zip(core_graph.values(), core_graph.keys()))
    trans_core.sort(reverse=True)
    # print(trans_core)
    core_seed = []
    for i in range(num):
        core_seed.append(trans_core[i][1])

    # betweenness
    betweenness_graph = nx.betweenness_centrality(graph)
    trans_betweenness = list(zip(betweenness_graph.values(), betweenness_graph.keys()))
    trans_betweenness.sort(reverse=True)
    # print(trans_betweenness)
    betweenness_seed = []
    for i in range(num):
        betweenness_seed.append(trans_betweenness[i][1])

    return degree_seed, core_seed, betweenness_seed


def count_r(graph, index):
    """传染病模型仿真达到稳定状态下R占整个网络的比例"""
    res_one = []
    res_more = []
    for a in range(1000):
        res_one.append(sir_one(graph, index))
        res_more.append(sir_more(graph, index))
    print("单接触SIR", sum(res_one) / len(res_one))
    print("全接触SIR", sum(res_more) / len(res_more))
    print("-----------------")


if __name__ == '__main__':
    graph = nx.read_edgelist(r"../dataset/kk.edgelist")

    degrees, cores, betweennesses = centrality_index(graph, 5)
    a=count_r(graph, 10)
    print(a)
    print("Degree Top 5: ", degrees)
    #count_r(G, 5)
