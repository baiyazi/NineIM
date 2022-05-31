"""
    函数功能：将原始数据集抽取edgelist到文件中
    数据集格式：
        #* --- paperTitle
        #@ --- Authors
        #t ---- Year
        #c  --- publication venue
        #index 00---- index id of this paper
        #% ---- the id of references of this paper (there are multiple lines, with each indicating a reference)
        #! --- Abstract
    example：
        1511035
        #*OQL[C++]: Extending C++ with an Object Query Capability.
        #@José A. Blakeley
        #t1995
        #cModern Database Systems
        #index0

        #*Transaction Management in Multidatabase Systems.
        #@Yuri Breitbart,Hector Garcia-Molina,Abraham Silberschatz
        #t1995
        #cModern Database Systems
        #index1

        #*Information geometry of U-Boost and Bregman divergence
        #@Noboru Murata,Takashi Takenouchi,Takafumi Kanamori,Shinto Eguchi
        #t2004
        #cNeural Computation
        #index436405
        #%94584
        #%282290
        #%605546
    数据集：./DBLP-citation-Jan8/DBLP-citation-Jan8.txt
    作者: 梦否
    时间：2021年2月21日 14:40:38
    函数名：getEdgeList
    参数：
        path="./DBLP-citation-Jan8/"
        filename="DBLP-citation-Jan8.txt"
"""
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

def getEdgeList(path="./DBLP-citation-Jan8/", filename="DBLP-citation-Jan8.txt"):
    with open(file=path+filename, mode="r", encoding="utf-8") as file:
        lines = file.readlines()
    current_node = 0
    edgeList = []
    for index, line in tqdm(enumerate(lines)):
        if index != 0:
            if line.startswith("#"):
                node_index_pattern = re.compile("\#index")
                if re.match(node_index_pattern, line) != None:
                    current_node = int(line[6:])
                node_refrence_pattern = re.compile("\#\%")
                if re.match(node_refrence_pattern, line) != None:
                    node_refrence = int(line[2:])
                    # 添加关系
                    edgeList.append([current_node, node_refrence])
    return edgeList


def save2File(edgeList, path="./DBLP-citation-Jan8/", filename="edgeList.csv"):
    edgeList = np.array(edgeList)
    node1 = list(edgeList[:, 0])  # 获取下标0列数据
    node2 = list(edgeList[:, 1])  # 获取下标1列数据
    # 存储数据到csv文件中
    df = pd.DataFrame({"node1": node1, "node2": node2})
    df.to_csv(path+filename, sep=",", index=False)  # index表示是否显示行名，default=True
    return "save to file success!"

if __name__ == '__main__':
    print(save2File(getEdgeList()))
