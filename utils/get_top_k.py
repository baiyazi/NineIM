import networkx as nx
import numpy as np
import pandas as pd
import heapq


def get_topk_by_corr(matrix, topk = 10):
    """
    计算两个向量之间的相关性来排序，得到topk
    :param matrix:
    :param topk:
    :return:
    """
    # print("network size after processing: ", len(matrix))
    column_lst = [str(e) for e in list(range(len(matrix)))]
    # 计算列表两两间的相关系数
    data_dict = {}  # 创建数据字典，为生成Dataframe做准备
    for col, gf_lst in zip(column_lst, matrix):
        data_dict[col] = gf_lst

    unstrtf_df = pd.DataFrame(data_dict)
    cor1 = unstrtf_df.corr()  # 计算相关系数，得到一个矩阵

    # 将一行或者每一列的累加，求平均值，然后作为该节点的score
    values = cor1.values
    results = list(np.sum(values, axis=0))

    _re = []
    for _, i in enumerate(column_lst):
        _re.append([i, results[_]])

    a = np.array(_re)
    b = a[:, 1]
    index = np.lexsort((b,))
    sorted_node = a[index][:, 0]
    return sorted_node[-topk:]


def get_topk_by_corr2(matrix, topk = 10):
    column_lst = [str(e) for e in list(range(len(matrix)))]
    print(len(column_lst))
    # 计算列表两两间的相关系数
    data_dict = {}  # 创建数据字典，为生成Dataframe做准备
    for col, gf_lst in zip(column_lst, matrix):
        data_dict[col] = gf_lst

    unstrtf_df = pd.DataFrame(data_dict)
    cor1 = unstrtf_df.T.corr()  # 计算相关系数，得到一个矩阵 , 按列求相关系数【df格式的数据，按照节点做关键字，就是列名】

    # 将一行或者每一列的累加，求平均值，然后作为该节点的score
    values = cor1.values
    resu = list(np.sum(values, axis=0))
    # 找每一行中的前topk个元素的下标
    _re = []
    for _, i in enumerate(column_lst):
        _re.append([i, resu[_]])

    a = np.array(_re)
    b = a[:, 1]
    index = np.lexsort((b,))
    sorted_node = a[index][:, 0]
    return sorted_node[-topk:]

# 尝试用另外的选择种子节点的方式
def get_topk_by_cosine(matrix, topk = 10):
    nodes = list(range(len(matrix)))
    matrix = np.array(matrix)
    matrix_dot = matrix.dot(matrix.T)
    resu = []
    for node_i in nodes:
        t = []
        for node_j in nodes:
            t.append(matrix_dot[node_i][node_j] / (np.linalg.norm(matrix[node_i]) * np.linalg.norm(matrix[node_j])))
        resu.append(t)
    # 找每一行中的前topk个元素的下标
    _indexs = []
    for row in resu:
        _index = heapq.nlargest(topk, range(len(row)), key=lambda x: row[x]) # 找出前topk个对应的下标
        _indexs.append(_index)
    # 统计_indexs中的下标的频率
    resus = {}
    for row in _indexs:
        for index in row:
            try:
                resus[index] += 1
            except:
                resus[index] = 0
    # 字典排序
    keys_ = list(resus.keys())
    keys_.sort(key=lambda x: resus[x], reverse=True)
    return [str(e) for e in keys_[:topk]]




