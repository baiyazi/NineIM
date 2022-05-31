"""
@Time: 2021/3/5 15:42
@version: ,
@author: ,
@description:
"""

import os
from gensim.models.word2vec import Word2Vec, LineSentence
from NINE import NineRW as nibe
from utils.classify import read_node_label, Classifier
from sklearn.linear_model import LogisticRegression
import numpy as np

from utils.get_top_k import get_topk_by_corr2
from utils.SIR import SIR
import time
import warnings
from setting import config

warnings.filterwarnings("ignore")


################################################################################
database_name = "dblp_500"  # 可选数据库：dblp_500 | football | CA-GrQc | netscience
################################################################################


random_walk_sequence_filename = "walks.csv"
database_template = "./data/{}/{}_graph.txt"  # graph的位置
label_template = "../data/{}/{}_labels.txt"  # label的位置

graph_label_filename = label_template.format(database_name, database_name)


def learn_model(walks, dimensions, window_size = 5):
    walks = [map(str, walk) for walk in walks]
    walks_filename = random_walk_sequence_filename
    for walk in walks:
        with open(walks_filename, mode="a", encoding='utf-8') as file:
            _line = ""
            for e in walk:
                if len(_line) != 0:
                    _line += " "
                _line += str(e)
            file.write(_line + "\n")
    sentences = LineSentence(walks_filename)
    model = Word2Vec(sentences, size=dimensions, window=window_size, min_count=0, sg=1, workers=1)

    if os.path.exists(walks_filename):
        os.remove(walks_filename)
    return model


def get_embeddings(model, graph):
    if model is None:
        print("model not train")
        return {}
    _embeddings = {}
    for word in graph.nodes():
        _embeddings[str(word)] = model.wv[str(word)]

    return _embeddings

def evaluate_embeddings(embeddings, tr_frac):
    print('-------------------')
    X, Y = read_node_label(graph_label_filename)
    # tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)

def evaluate(embeddings_):
    """
    F1评估
    :param embeddings_:
    :return:
    """
    if database_name == "test":
        evaluate_embeddings(embeddings_, 0.6)
    else:
        pro = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for p in pro:
            evaluate_embeddings(embeddings_, p)


def getFinalEmbedding(databaseName, args):

    RW = nibe.Graph(databaseName, args.beta, args.global_random_parameter, args.local_random_parameter)
    walks = RW.simulate_walks(args.walk_length)
    model = learn_model(walks, args.dimensions, args.window_size)
    embeddings = get_embeddings(model, RW.g)
    nodes = list(RW.g.nodes)
    nodes.sort()
    embeddings_ = []
    for node in nodes:
        embeddings_.append(embeddings[str(node)])
    embeddings_ = np.array(embeddings_)
    return embeddings_



if __name__ == "__main__":
    args = config.parse_args()
    embeddings_ = getFinalEmbedding(database_name, args)
    # 节点分类评估
    print(embeddings_)
    evaluate(embeddings_)

