"""
    函数功能：从edgeList.csv中进行随机抽取关系，构成子图
    数据集：./DBLP-citation-Jan8/edgeList.csv
    example:
        node1,node2
        25,165
        368,832280
        368,574798
        373,145
        373,702
    作者：梦否
    时间：2021年2月21日 15:28:57
    函数名：getSubGraph
    参数：
        path="./DBLP-citation-Jan8/"
        filename="edgeList.csv"
"""
import pandas as pd
import numpy as np
import random
import networkx as nx
from sklearn.utils import shuffle
from src.utils.community_detection import communityPartition
from src.utils.utils import read_graph



def getSubGraph(path="./DBLP-citation-Jan8/", filename="edgeList.csv", graph_size = 500):
    df = pd.read_csv(filepath_or_buffer=path+filename, sep=",")
    # 1. 找到最大元素
    max_1 = np.max(list(df.node1))
    max_2 = np.max(list(df.node2))
    min_1 = np.min(list(df.node1))
    min_2 = np.min(list(df.node2))
    max_node =  max_1 if max_1 > max_2 else max_2
    min_node =  min_1 if min_1 > min_2 else min_2
    # 2. 随机选择300个节点
    _list = list(range(min_node, max_node))
    random.shuffle(_list)
    nodes = random.sample(_list, graph_size)
    # 3. 删除df.node1列对应的行，没选择中的元素
    results = []
    for row in df.itertuples():
        node1 = int(getattr(row, 'node1'))
        node2 = getattr(row, 'node2')
        if node1 in nodes:
            results.append([node1, node2])
    # 存储函数
    def save2File(edgeList, filename):
        edgeList = np.array(edgeList)
        node1 = list(edgeList[:, 0])  # 获取下标0列数据
        node2 = list(edgeList[:, 1])  # 获取下标1列数据
        # 将节点重新编号
        nodes = set(node1).union(set(node2))
        print("max node label is " + str(len(nodes)))
        map_ = dict()
        for index, node in enumerate(nodes):
            map_[node] = index + 1
        node1 = [map_[e] for e in node1]
        node2 = [map_[e] for e in node2]
        # 存储数据到csv文件中
        df = pd.DataFrame({"node1": node1, "node2": node2})
        df.to_csv(filename, sep=",", index=False, header=None)  # index表示是否显示行名，default=True
        print("save graph to file success!")
    # 存储 edgelist
    filename="dblp_500_graph.txt"
    save2File(results, filename)

    # 5. 重新读取到nx_G
    G = read_graph(filename, False)
    # 6. 划分社区
    com_dict = communityPartition(G)
    # 7. 存储为label
    filename = "dblp_500_labels.txt"
    labels = []
    for key in com_dict.keys():
        for node in com_dict[key]:
            labels.append([node, key])

    labels = np.array(labels)
    df = pd.DataFrame({"node1": labels[:, 0], "label": labels[:, 1]})
    # 一定是打乱顺序再存储
    df = shuffle(df)
    df.to_csv(filename, sep=",", header=None, index=False)
    print("Storage graph label successful!")


# def drawGraph():
#     g = nx.read_edgelist("./DBLP-citation-Jan8/temp.edgelist")
#     nodes = list(g.nodes)
#     # 统计节点的度分布
#     degrees = dict()
#     for node in nodes:
#         degree = g.degree(node)
#         try:
#             degrees[degree] += 1
#         except:
#             degrees[degree] = 1
#     # 绘制幂律分布图像
#     # x 个数； y 度大小
#     plt.scatter(degrees.keys(), [math.log(int(ele)) for ele in list(degrees.values())])
#     plt.show()





if __name__ == '__main__':
    getSubGraph()
