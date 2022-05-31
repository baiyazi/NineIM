

import matplotlib.pyplot as plt
import numpy as np
from utils.classify import read_node_label
from sklearn.manifold import TSNE


def plot_embeddings(embeddings, filename):
    """
    绘制2维图像
    :param embeddings:
    :return:
    """
    X, Y = read_node_label(filename)
    emb_list = []
    for k in X:
        emb_list.append(embeddings[int(k)])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1],
                    label=c)  # c=node_colors)
    #plt.legend()
    plt.show()