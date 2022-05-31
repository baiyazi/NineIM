#!/usr/bin/env python
# coding: utf-8

# In[2]:


import networkx as nx
import numpy as np
import community as community_louvain
import pandas as pd
from sklearn.utils import shuffle
import random


# In[3]:


def save_labels(partition):
    labels = []
    for key in partition.keys():
        labels.append([key, partition[key]])
    labels = np.array(labels)
    df = pd.DataFrame({"node1": labels[:, 0], "node2": labels[:, 1]})
    df = shuffle(df)
    df.to_csv("netscience_labels.txt", sep=" ", header=None, index=False)


# In[4]:


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


# In[5]:


G = nx.read_gml("netscience.gml")
G = nx.to_undirected(G)

# 2. 重新编号
node_map = {}
for _, node in enumerate(G):
    node_map[node] = _

# 找到最大的联通子图，然后和孤立节点添加连接边
largest_cc = list(max(nx.connected_components(G), key=len))
components = list(nx.connected_components(G))


relationship = []
for node in G.nodes:
    for n in G.neighbors(node):
        relationship.append([node_map[node], node_map[n]])


for component in components:
    component = list(component)
    flag = False
    for node in component:
        if node in largest_cc:
            flag = True
    if not flag:
        relationship.append([node_map[random.choice(component)], node_map[random.choice(largest_cc)]])


relationship = np.array(relationship)

# 4. 存储
filename = "netscience_graph.txt"
df = pd.DataFrame({"node1":relationship[:, 0], "node2": relationship[:, 1]})
df.to_csv(filename, sep=",", header=None, index=False)
print("Storage graph data successful!")


# 5. 重新读取到nx_G
G = read_graph(filename, False)
# 6. 划分社区
partition = community_louvain.best_partition(G)
save_labels(partition)

