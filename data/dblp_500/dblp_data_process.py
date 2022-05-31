#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import  os


# In[2]:
# 读取的文件
graph_file_name = "./dblp/com-dblp.ungraph.txt"
community_file_name = "./dblp/com-dblp.all.cmty.txt"
# 子图抽取的保存文件
sub_graph_file_name = "dblp_500_graph.txt"
sub_community_file_name = "dblp_500_labels.txt"



def bubble_sort(key, value):
    """
    简单冒泡排序
    :param key:
    :param value:
    :return:
    """
    for idx_i, i in enumerate(key):
        for idx_j, j in enumerate(key):
            if i > j:
                temp = key[idx_i]
                key[idx_i] = key[idx_j]
                key[idx_j] = temp

                # value
                temp = value[idx_i]
                value[idx_i] = value[idx_j]
                value[idx_j] = temp
    return value


# In[7]:


def read_node_community_from_file(cm_filename):
    communitys = {}
    with open(cm_filename, mode="r", encoding="utf=8") as file:
        lines = file.readlines()

    each_community_len_list = []
    for _idx, line in enumerate(lines):
        line = line.rsplit()
        communitys[str(_idx)] = line
        each_community_len_list.append(len(line))
    return communitys, each_community_len_list, len(lines)


# In[4]:


def get_top_k_community(communitys, community_number, each_community_len_list, randomK=7):
    """
    按照社区大小排序, 返回选择的节点集合，选定的社区列表
    :param communitys:
    :param community_number:
    :param each_community_len_list:
    :param topk:
    :return:
    """
    import random
    com_idxs = list(range(len(each_community_len_list)))
    com_idxs = bubble_sort(each_community_len_list, com_idxs)
    selected_nodes = []
    used_community_identity = []
    count = 0
    for com_idx in com_idxs:
        if random.random() < 0.004:
            selected_nodes += [e.strip() for e in communitys[str(com_idx)]]
            count += 1
            used_community_identity.append(str(com_idx))
            if count == randomK:
                break
    selected_nodes = list(set(selected_nodes))
    selected_nodes.sort()
    return selected_nodes, [communitys[e] for e in used_community_identity]


# In[9]:


communitys, each_community_len_list, community_number = read_node_community_from_file(community_file_name)


# In[10]:


selected_nodes, used_communitys = get_top_k_community(communitys, community_number, each_community_len_list)


# In[14]:


import networkx as nx
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
        for line in f:
            u, v = map(int, line.split(","))
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u, v, weight=1)
    if not directed:
        G = G.to_undirected()
    return G


# In[41]:


g = read_graph(graph_file_name, False)
nodes = [int(e) for e in selected_nodes]
print("当前抽取网络的节点个数为：", len(nodes))
# 节点映射到新的标签
node_map = {}
_id = 0
for node in nodes:
    node_map[node] = _id
    _id += 1
# 子图保存
subG = nx.Graph.subgraph(g, nodes)
nx.write_edgelist(subG, "sub_graph.edgelist", delimiter=',')
df = pd.read_csv("sub_graph.edgelist")
os.remove("sub_graph.edgelist")

new_nodes = []
for row in df.values:
    new_nodes.append([node_map[row[0]], node_map[row[1]]])
new_nodes = np.array(new_nodes)

df = pd.DataFrame({'node1': new_nodes[:, 0], 'node2': new_nodes[:, 1]})
df.to_csv(sub_graph_file_name, sep=",", index=False)

with open(sub_community_file_name, encoding="utf-8", mode="w") as file:
    for com in used_communitys:
        line = ""
        for node in com:
            if line != "":
                line += "\t"
            line += node
        line += "\n"
        file.write(line)


# In[26]:
print("网络抽取完成")



"""
>>> G=nx.path_graph(4)
>>> nx.write_edgelist(G, "test.edgelist")
>>> G=nx.path_graph(4)
>>> fh=open("test.edgelist",'wb')
>>> nx.write_edgelist(G, fh)
>>> nx.write_edgelist(G, "test.edgelist.gz")
>>> nx.write_edgelist(G, "test.edgelist.gz", data=False)
"""


# In[ ]:




