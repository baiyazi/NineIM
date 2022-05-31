"""
@Time: 2021/3/5 15:42
@version: ,
@author: ,
@description:
"""
import networkx as nx

from utils.get_top_k import get_topk_by_corr, get_topk_by_cosine, get_topk_by_corr2
from utils.SIR01 import sir_one, sir_more
from utils.SIR import SIR
import time
import warnings


warnings.filterwarnings("ignore")


database_template = "data/{}/{}_graph.txt"  # graph的位置



def select_seed_and_diffusion(g, embeddings_, topk, beta, gama, bunch_size=20):
    """
    选择种子节点，并进行SIR传播，bunch_size是SIR模型模拟的次数，默认为20
    :param embeddings_:
    :param topk:
    :param bunch_size:
    :return:
    """
    # seeds = list(get_topk_by_corr(embeddings_, topk))
    seeds = list(get_topk_by_corr2(embeddings_, topk)) # (dim, node_number)
    # seeds = list(get_topk_by_cosine(embeddings_, topk))
    r_state_nodes = 0
    total_time = 0
    for i in range(bunch_size):
        start_time = time.time()
        # r_state_nodes += sir_more(g, seeds, beta, gama)
        r_state_nodes += SIR(g, seeds, beta, gama).run()
        end_time = time.time()
        total_time += (end_time - start_time)
    return r_state_nodes / bunch_size, total_time / bunch_size


def influence_measure(database_name, beta = 0.07, gama = 9e-2, embeddings_=None):
    with open("result_temp/{}_beta_{}_NineIM.csv".format(database_name, beta), mode="a", encoding="utf-8") as file:
        number_of_k = [2, 4, 5, 10, 15, 20, 25, 30, 35, 40, 40, 50]
        line_nodes = "NineIM,"
        for k in number_of_k:
            g = nx.read_edgelist(database_template.format(database_name, database_name), delimiter=",", nodetype=str)
            r_state_nodes, times = select_seed_and_diffusion(g, embeddings_, k, beta, gama, bunch_size=100)
            line_nodes += str(r_state_nodes)
            if k != number_of_k[-1]:
                line_nodes += ","
            else:
                line_nodes += ","
        file.write(line_nodes+"\n")



