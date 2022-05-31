import networkx as nx
import numpy as np
import community as community_louvain
import pandas as pd
import argparse
from sklearn.utils import shuffle
from src.utils.utils import read_graph

def save_labels(partition):
    labels = []
    for key in partition.keys():
        labels.append([key, partition[key]])
    labels = np.array(labels)
    df = pd.DataFrame({"node1": labels[:, 0], "node2": labels[:, 1]})
    df = shuffle(df)
    df.to_csv("netscience_labels.txt", sep=" ", header=None, index=False)

def load_gml():
    G = nx.read_gml("netscience.gml")
    G = nx.to_undirected(G)
    # 找到最大联通子图
    largest_cc = max(nx.connected_components(G), key=len)
    print("Number of isolated nodes is ", len(list(G.nodes)) - len(largest_cc))

    # 2. 重新编号
    node_map = {}
    for _, node in enumerate(largest_cc):
        node_map[node] = _
    # 3. 保存关系
    relationship = []
    for node in largest_cc:
        for n in G.neighbors(node):
            relationship.append([node_map[node], node_map[n]])

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


#
# def parse_args():
#     parser = argparse.ArgumentParser(description="Run NIBE(Node Influence Based Embedding) program.")
#     parser.add_argument('--weighted', type=bool, default=False, help='Is the input graph weighted?')
#     parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')
#     parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')
#     return parser.parse_args()
#
#
# def save_matrix(matrix, database_name):
#     print("saving matrix...")
#     np.save(database_name+"_inf_matrix", matrix)
#     print("save matrix successful!")

if __name__ == "__main__":
    load_gml()

    # args = parse_args()
    # database_name = "dblp_500" # dblp_500
    # beta = 0.4
    # global_random_parameter = 0.3
    # local_random_parameter = 0.2
    #
    # nibe = nibe.Graph("../dataset_process/"+database_name, beta, global_random_parameter, local_random_parameter)
    # matrix= nibe.get_graph_measurement_matrix()
    # print(matrix.shape)
    # save_matrix(matrix, database_name)


