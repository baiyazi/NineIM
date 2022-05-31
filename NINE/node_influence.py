"""
    时间：2021年2月3日
    作者：梦否
"""

from NINE.NineRW import Graph
from utils.utils import normalization_operation
import networkx as nx


def get_our_node_influence_measure(g, global_influence_parameter = 0.3, local_influence_parameter = 0.2, round_k=4):
    """
    返回图中每个节点自身的重要性程度
    【计算方式】：
        依次计算该节点对其邻居节点的影响力大小（包括全局影响力和局部影响力），然后累加求和
    :param g:
    :return:
    """
    nibe = Graph(g, 0, global_influence_parameter, local_influence_parameter) # beta, 是涉及到随机游走的参数，这里在计算节点influence时候没什么用
    infs = []
    nodes = list(g.nodes)
    # 对元素节点排序
    nodes.sort()
    for node in nodes:
        neighbors = list(g.neighbors(node))
        _sum = 0.0
        for n in neighbors:
            i_g = nibe.get_node_global_measurement(node, n)
            i_l = nibe.get_node_local_measurement(node, n)
            i = i_g * global_influence_parameter + i_l * local_influence_parameter
            _sum += i
        infs.append(_sum / len(neighbors))
    return normalization_operation(infs, round_k)


def get_degree_influence_measure(g, round_k=4):
    """
    节点度的度量
    :param g:
    :return:
    """
    nodes = list(g.nodes)
    # 对元素节点排序
    nodes.sort()
    infs = []
    for node in nodes:
        neighbors = list(g.neighbors(node))
        infs.append(len(neighbors))
    return normalization_operation(infs, round_k)


def get_closeness_centrality(g, round_k=4):
    """
    closeness_centrality
    :param g:
    :return:
    """
    nodes = list(g.nodes)
    # 对元素节点排序
    nodes.sort()
    infs = []
    cc = nx.closeness_centrality(g)
    for node in nodes:
        infs.append(cc[node])
    return normalization_operation(infs, round_k)

def get_betweenness_centrality(g, round_k=4):
    nodes = list(g.nodes)
    # 对元素节点排序
    nodes.sort()
    infs = []
    cc = nx.betweenness_centrality(g)
    for node in nodes:
        infs.append(cc[node])
    return normalization_operation(infs, round_k)

def get_eigenvector_centrality(g, round_k=4):
    nodes = list(g.nodes)
    # 对元素节点排序
    nodes.sort()
    infs = []
    cc = nx.eigenvector_centrality(g)
    for node in nodes:
        infs.append(cc[node])
    return normalization_operation(infs, round_k)






