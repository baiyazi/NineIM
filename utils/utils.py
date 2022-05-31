"""
    时间：2021年3月3日
    作者：梦否
"""

import networkx as nx
import numpy as np


def read_graph(edgelist_file, directed):
    """
    读取csv文件到networkx的图G中
    调用案例：
        g = read_graph("../data/edgelist.csv", False)  # 无向图
    csv格式：

    :param edgelist_file:
    :param directed:
    :return:
    """
    # 定义一个Graph来存储节点边等信息，默认初始权重为1
    G = nx.DiGraph()

    with open(edgelist_file, mode="r", encoding="utf-8") as f:
        # 第一行存储的是顶点的数目和边的数目
        n, m = f.readline().split(",")
        for line in f:
            u, v = map(int, line.split(","))
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u, v, weight=1)
    if not directed:
        G = G.to_undirected()
    return G

def normalization_operation(_matrix, axis=1, round_k=4):
    if type(np.array([])) != type(_matrix):
        _matrix = np.array(_matrix)
    shape = _matrix.shape
    _len = len(_matrix)
    try:
        shape[1]
        dimension = 2
    except:
        dimension = 1
    if dimension == 0:
        return
    if dimension == 2:
        if axis == 0:
            # 矩阵按列归一化
            max_vals = np.max(_matrix, axis=0)
            min_vals = np.min(_matrix, axis=0)
        else:
            # 矩阵按行归一化
            max_vals = np.max(_matrix, axis=1)
            min_vals = np.min(_matrix, axis=1)
        _range = max_vals - min_vals
        print(_range)
        result = []
        for index in range(_len):
            col = _matrix[:, index]
            result.append((col - min_vals[index]) / (max_vals[index] - min_vals[index]))
        return result
    else:
        # 维度为1，这里直接返回比例值
        _sum = 0.0
        for val in _matrix:
            _sum += val
        return [round(e, round_k) for e in _matrix / _sum]




# if __name__ == '__main__':
#     g = read_graph("../data/test.edgelist", False)
#     for node in g.nodes:
#         print(node, "degree is", g.degree(node))
