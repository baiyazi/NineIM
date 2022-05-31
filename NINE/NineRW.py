"""
    node influence based embedding
    时间：2020年12月2日
    作者：梦否
"""

import random
import numpy as np
import networkx as nx
import math
from utils.alias import create_alias_table, alias_sample
from utils.community_detection import communityPartition
import os
from . import calc_matrix
from tqdm import tqdm


class Graph:
    """
        调用walk即可
    """
    def __init__(self, database_name, beta, global_influence_parameter, local_influence_parameter, is_new_measure_matrix=True):
        """
        初始化
        :param G_nx:  图
        :param beta:  超参数，控制图是从社区中随机选择还是考虑概率矩阵的有偏随机游走
        :param global_influence_parameter:  超参数，节点值计算全局值所占比重
        :param local_influence_parameter:  超参数，节点值计算局部值所占比重
        """
        self.beta = beta
        self.g = nx.read_edgelist("../data/"+database_name+"/"+database_name+"_graph.txt", delimiter=",", nodetype=int)
        self.nodes = list(self.g.nodes)
        self.N = len(self.nodes)
        self.g_p = global_influence_parameter
        self.l_p = local_influence_parameter
        self.initial()
        self.database_name = database_name
        self.alpha = 0.01
        self.is_new_measure_matrix = is_new_measure_matrix
        pass

    def initial(self):
        self.graph_has_weight()
        self.graph_total_degree = None
        self.idx2node, self.node2idx = self.preprocess_nxgraph()
        self._gen_sampling_table()
        self.graph_measurement_matrix = None
        self.node_community_map = None
        self.community_node_map = None

    def preprocess_nxgraph(self):
        node2idx = {}
        idx2node = []
        node_size = 0
        for node in self.g.nodes():
            node2idx[node] = node_size
            idx2node.append(node)
            node_size += 1
        return idx2node, node2idx

    def graph_has_weight(self):
        """
        判断输入的图是否含有权重weight
        """
        try:
            self.g.nodes[0]['weight']
            self.is_weight_graph = True
        except:
            self.is_weight_graph = False

    def get_nodes_pairs_digkstra_path_list(self, start_node, end_node, max_len = 15):
        """
        获取图中所有点到其余点的狄杰斯特拉最短距离
        :param start_node: 开始节点
        :param end_node: 终止节点
        :param max_len:  最长计算到多少，因为太长就没有意义了
        :return:  最短距离列表。 空列表表示最短距离超过了这里定义的max_len
        """
        if self.graph_all_min_path_dict == None:
            if self.is_weight_graph:
                self.graph_all_min_path_dict = nx.all_pairs_dijkstra_path(self.g, cutoff=max_len, weight="weight")
            else:
                self.graph_all_min_path_dict = nx.all_pairs_dijkstra_path(self.g, cutoff=max_len)
        try:
            _list = self.graph_all_min_path_dict[start_node][end_node]
        except:
            _list = []
        return _list

    def random_select_from_com(self, node):
        """
        社区划分，并增加社区-节点，节点-社区的映射，然后从节点社区中随机选择一个
        :param node:
        :return:
        """
        if self.node_community_map == None:
            self.node_community_map = dict()
            _dict = communityPartition(self.g)
            self.community_node_map = _dict
            for key in _dict.keys():
                for node in _dict[key]:
                    self.node_community_map[node] = key
        # 从社区中随机选择一个
        return random.choice(self.community_node_map[self.node_community_map[node]])

    def get_average_min_path_len(self, node_i, node_j, removed=False):
        """
        当前节点到除了remove_node节点之外的所有节点的最短路径长度的平均值
        :param current_node:
        :param remove_node:
        :return:
        """
        if node_j == None or node_i == None:
            raise Exception("the parameter of node can't be None!")
        if removed:
            n = self.N
        else:
            n = self.N - 1
        # current_node到所有节点的最短路径
        nodes = self.nodes
        g = self.g
        _len = 0
        for node in nodes:
            if node != node_i:
                if self.is_weight_graph:
                    try:
                        temp_len = nx.dijkstra_path_length(g, node_j, node, weight='weight')
                    except:
                        # temp_len 应该是unreachable的值，这里直接定义为0
                        temp_len = 0
                    _len += temp_len
                else:
                    try:
                        temp_len = nx.dijkstra_path_length(g, node_j, node)
                    except:
                        # temp_len 应该是unreachable的值，这里直接定义为0
                        temp_len = 0
                    _len += temp_len
        return _len / n


    def remove_and_save_node_edges_informatioin(self, node_i):
        """
        计算dijkstra最短路径的时候，需要先移除这个节点
        :param node:
        :return: 当前被移除节点和它的邻居节点列表
        """
        g = self.g
        node_neighbors = list(g.neighbors(node_i))
        g.remove_node(node_i)
        self.g = g
        self.nodes = g.nodes
        self.N = len(list(self.g.nodes))
        return node_i, node_neighbors

    def reshape_node_edges_information(self, node, node_neighbors):
        """
        重新将移除节点node添加到图中，并添加相应的边信息
        :param node:
        :param node_neighbors:
        :return:
        """
        g = self.g
        g.add_node(node)
        for n in node_neighbors:
            g.add_edge(node, n)
        self.g = g
        self.nodes = g.nodes
        self.N = len(list(self.g.nodes))

    def get_graph_efficiency(self, node_i, node_j, remove=False):
        """
        网络的网络效率。
        :param remove_node:  待移除节点
        :return:  网络的网络效率值
        """
        if self.g.degree(node_i) == 1:
            return 0
        if self.g.degree(node_i) == 2:
            ns = list(self.g.neighbors(node_i))
            if self.g.has_edge(ns[0], ns[1]):
                return 0
        if not remove:
            avg_len = self.get_average_min_path_len(node_i, node_j, removed=False)
        else:
            node, node_neighbors = self.remove_and_save_node_edges_informatioin(node_i)
            avg_len = self.get_average_min_path_len(node_i, node_j, removed=True)
            self.reshape_node_edges_information(node, node_neighbors)
        return avg_len

    def get_node_global_measurement(self, node_i, node_j):
        """
        【节点的全局重要性度量】
        通过计算移除节点及其连边前后网络效率的变化率来衡量节点重要性。
        :param node: 当前待判断节点
        :return: 当前node的效率中心性值
        """
        graph_efficiency = self.get_graph_efficiency(node_i, node_j, remove=False)
        if graph_efficiency == 0:
            return 0
        return math.fabs((graph_efficiency - self.get_graph_efficiency(node_i, node_j, remove=True))) / graph_efficiency

    def get_one_hop_val(self, node):
        """
        得到直接1跳邻居的度量，抽取出来只是方便循环调用
        :param node:
        :return:
        """
        g = self.g
        nodes = self.nodes
        if self.graph_total_degree == None:
            val = 0.0
            for n in nodes:
                val += g.degree(n)
            self.graph_total_degree = val
        one_hop_val = 1. * g.degree(node) / self.graph_total_degree
        return one_hop_val

    def get_node_local_measurement(self, node_i, node_j):
        """
        获取当前节点node的本地局部重要性度量
        【直接邻居+2 hop 邻居】
        直接邻居节点和次邻居节点的得分值之和
        :param node: 待计算节点
        :return: 该节点的本地局部重要度度量
        """
        g = self.g
        neighbors_i = set(list(g.neighbors(node_i)))
        neighbors_j = set(list(g.neighbors(node_j)))
        if len(neighbors_j) == 0 or len(neighbors_i) == 0:
            return 0
        jaccard_coefficient = float(len(neighbors_i & neighbors_j)) / len(neighbors_i | neighbors_j)
        return self.get_one_hop_val(node_i) + jaccard_coefficient

    def get_node_important_measurement(self, node_i, node_j):
        """
        综合节点的全局度量&局部度量值得到节点的综合度量值
        :param node:
        :return:
        """
        return self.g_p * self.get_node_global_measurement(node_i, node_j) +\
            self.l_p * self.get_node_local_measurement(node_i, node_j)

    def get_graph_measurement_matrix(self):
        if not self.is_new_measure_matrix:
            name = "../dataset_process/"+self.database_name+"_inf_matrix.npy"
            print("loading the preprocess matrix. {}".format(name))

            if os.path.isfile(name):
                print(name)
                # load matrix
                return np.load(name)

            # 初始化一个矩阵NxN
            n = self.N
            nodes = self.nodes
            nodes.sort()
            _matrix = [[0] * (n) for i in range(n)]

            for node_i in tqdm(nodes):
                for node_j in nodes:
                    print(node_i, "->", node_j)
                    if node_i != node_j:
                        _matrix[node_i][node_j] = round(self.get_node_important_measurement(node_i, node_j), 4)

            return np.array(_matrix)

        else:
            name = "../dataset_process/"+self.database_name+"_new_inf_matrix.npy"
            print("training the node influence matrix. {}".format(name))

            if os.path.isfile(name):
                print(name)
                # load matrix
                return np.load(name)

            _matrix = calc_matrix.get_graph_measure_matrix(self.g, self.database_name)

            return np.array(_matrix)

    def calc_transition_probs(self):
        #return self.normalization_operation(self.get_graph_measurement_matrix())
        return self.get_graph_measurement_matrix()

    def _gen_sampling_table(self):

        # create sampling table for vertex
        power = 0.75
        numNodes = self.N
        node_degree = np.zeros(numNodes)  # out degree
        node2idx = self.node2idx

        for edge in self.g.edges():
            node_degree[node2idx[edge[0]]] += self.g[edge[0]][edge[1]].get('weight', 1.0)

        total_sum = sum([math.pow(node_degree[i], power)
                         for i in range(numNodes)])
        norm_prob = [float(math.pow(node_degree[j], power)) /
                     total_sum for j in range(numNodes)]
        self.node_accept, self.node_alias = create_alias_table(norm_prob)

        # create sampling table for edge
        numEdges = self.g.number_of_edges()
        total_sum = sum([self.g[edge[0]][edge[1]].get('weight', 1.0)
                         for edge in self.g.edges()])
        norm_prob = [self.g[edge[0]][edge[1]].get('weight', 1.0) *
                     numEdges / total_sum for edge in self.g.edges()]

        self.edge_accept, self.edge_alias = create_alias_table(norm_prob)

    def _single_walk_2(self, walk_length, start_node):
            walk = [start_node]
            for _ in range(walk_length - 1):
                cur = walk[-1]
                cur_nbrs = [node for node in self.g.neighbors(cur)]
                if len(cur_nbrs) > 0:
                    v = alias_sample(self.node_accept, self.node_alias)
                    walk.append(v) # 简化写法：walk = walk + random.sample(cur_nbrs, 1)
                    if random.random() < self.getInfluence(v, cur):
                        walk.append(cur)
                else:
                    self.random_select_from_com(cur)

            walk = [str(w) for w in walk]
            return walk

    def _single_walk(self, walk_length, start_node):
        walk = [start_node]
        for _ in range(walk_length - 1):
            cur = walk[-1]
            cur_nbrs = [node for node in self.g.neighbors(walk[-1])]
            if len(cur_nbrs) > 0:
                #v = alias_sample(self.node_accept, self.node_alias)
                v = random.sample(cur_nbrs, 1)[0]
                walk.append(v)
                if random.random() < 1 - self.getInfluence(v, cur):
                    walk.append(cur)
            else:
                if random.random() < self.alpha:
                    self.random_select_from_com(cur)

        walk = [str(w) for w in walk]
        return walk

    def simulate_walks(self, walk_length):
        walks = []
        walk_number = 10
        for node in self.g.nodes():
            for _ in range(walk_number):
                walk_from_node = self._single_walk(walk_length, node)
                walks.append(walk_from_node)
        return walks

    def getInfluence(self, node_i, node_j):
        """
        对于numpy格式的数据，如果用np.array() == None  会返回np.array([Boolean])所以这里判断类型
        :param node_i:
        :param node_j:
        :return:
        """
        if type(self.graph_measurement_matrix) == type(None):
            self.graph_measurement_matrix = self.get_graph_measurement_matrix()
        return self.graph_measurement_matrix[node_i][node_j]
