from utils.utils import read_graph
import numpy as np
import random
import networkx as nx
import tqdm


def get_matrix_s(g, matrix_a):
    degrees = []
    nodes = list(g.nodes)
    nodes.sort()
    for node in nodes:
        degrees.append(g.degree(node))
    degrees = 1 / np.array(degrees)
    matrix_d = np.diag(degrees)
    matrix_s = matrix_d.dot(matrix_a)
    return matrix_s


def get_matrix_one_hot(g):
    val = [1 for e in list(g.nodes)]
    matrix_one_hot = np.diag(val)
    return matrix_one_hot


def update_p_k(alpha, p_k_1, matrix_s):
    return alpha * np.dot(p_k_1, matrix_s)


def get_p_k(matrix_s, matrix_one_hot, node_i, isStartWithZero):
    alpha = 0.7
    k = 2
    if not isStartWithZero:
        orginal = matrix_one_hot[int(node_i) - 1].copy()
        orginal_update = matrix_one_hot[int(node_i) - 1].copy()
    else:
        orginal = matrix_one_hot[int(node_i)].copy()
        orginal_update = matrix_one_hot[int(node_i)].copy()
    for i in range(k):
        if random.random() < alpha:
            orginal_update = update_p_k(alpha, orginal_update, matrix_s)
        else:
            orginal_update = orginal
    return orginal_update


def get_normal(_matrix, axis=1):
    if type(np.array([])) != type(_matrix):
        _matrix = np.array(_matrix)
    _len = len(_matrix)
    result = []
    if axis == 0:
        for index in range(_len):
            col = _matrix[:, index]
            if np.sum(col) == 0:
                result.append(col)
            else:
                result.append([round(e, 4) for e in list(col / np.sum(col))])
        return np.array(result).T
    else:
        assert axis == 1
        for index in range(_len):
            col = _matrix[index, :]
            if np.sum(col) == 0:
                result.append(col)
            else:
                result.append([round(e, 4) for e in list(col / np.sum(col))])
        return np.array(result)


def get_matrix_b(matrix_m):
    if type(np.array([])) != type(matrix_m):
        matrix_m = np.array(matrix_m)
    temp_matrix_m = []
    for row in matrix_m:
        temp_row = []
        for e in row:
            if e == 0:
                temp_row.append(0.0)
            else:
                temp_row.append(1 / e)
        temp_matrix_m.append(temp_row)
    matrix_b = get_normal(temp_matrix_m, axis=1)
    return np.array(matrix_b)


def get_our_node_influence_measure(g, global_influence_parameter = 0.3, local_influence_parameter = 0.2, isStartWithZero=False):
    matrix_a = nx.adj_matrix(g).toarray()
    matrix_s = get_matrix_s(g, matrix_a)
    matrix_one_hot = get_matrix_one_hot(g)
    matrix_m = []
    nodes = list(g.nodes)
    nodes.sort()
    for node in g.nodes:
        re = get_p_k(matrix_s, matrix_one_hot, node, isStartWithZero)
        matrix_m.append(re)

    #matrix_b = get_matrix_b(matrix_m)
    matrix_b = matrix_m

    infs = []
    nodes = list(g.nodes)
    # 对元素节点排序
    nodes.sort()
    for node in tqdm.tqdm(nodes):
        _temp = []
        for n in nodes:
            if not isStartWithZero:
                i_g = matrix_b[node-1][n-1]
            else:
                i_g = matrix_b[node][n]
            i_l = get_node_local_measurement(g, node, n)
            _temp.append(round(i_g * global_influence_parameter + i_l * local_influence_parameter, 4))
        infs.append(_temp)
    # return np.array(get_normal(infs, axis=1))
    return np.array(infs)

def get_one_hop_val(g, node):
    """
    得到直接1跳邻居的度量，抽取出来只是方便循环调用
    :param node:
    :return:
    """
    nodes = g.nodes
    graph_total_degree = 0
    for n in nodes:
        graph_total_degree += g.degree(n)
    one_hop_val = 1. * g.degree(node) / graph_total_degree
    return one_hop_val

def get_node_local_measurement(g, node_i, node_j):
    """
    获取当前节点node的本地局部重要性度量
    【直接邻居+2 hop 邻居】
    直接邻居节点和次邻居节点的得分值之和
    :param node: 待计算节点
    :return: 该节点的本地局部重要度度量
    """
    neighbors_i = set(list(g.neighbors(node_i)))
    neighbors_j = set(list(g.neighbors(node_j)))
    if len(neighbors_j) == 0 or len(neighbors_i) == 0:
        return 0
    jaccard_coefficient = float(len(neighbors_i & neighbors_j)) / len(neighbors_i | neighbors_j)
    return get_one_hop_val(g, node_i) + jaccard_coefficient



#################################################################
# database_name = "netscience"  # 可选数据库：dblp_500 | football | netscience
##################################################################


def get_graph_measure_matrix(g, database_name):
    storage_matrix_temple = "../dataset_process/{}_new_inf_matrix"
    print("Generate node influence matrix...")
    influence_matrix = get_our_node_influence_measure(g, isStartWithZero=True)
    np.save(storage_matrix_temple.format(database_name), influence_matrix)
    print("Storage node influence matrix in ", storage_matrix_temple.format(database_name))
    return influence_matrix

# test
if __name__ == '__main__':
    database_name = "case"
    g = nx.read_edgelist("../data/"+database_name+"/"+database_name+"_graph.txt", delimiter=",", nodetype=int)
    influence_matrix = get_graph_measure_matrix(g, database_name)
    print(influence_matrix)



