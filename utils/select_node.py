import numpy as np
import heapq

def select_node(embeddings, topk=10):
    """
    按照《Social Influence Maximization for Public Health Campaigns》提出的种子节点筛选方法
        ① 计算xxT
        ② 计算每一行的元素的值的平均值，该值作为score
    :param embeddings:
    :param k:
    :return:
    """
    x = []
    _len = 0
    keys = list(embeddings.keys())
    for key in keys:
        _len += 1
        x.append(embeddings[key])
    x = np.array(x)
    k = x.dot(x.T)
    p = k.sum(axis=1) / _len
    resu = bubble_sort(p, keys)
    return resu[:topk]

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


