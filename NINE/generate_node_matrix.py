
import networkx as nx
from NINE.calc_matrix import get_graph_measure_matrix

#################################################################
database_name = "netscience"  # 可选数据库：dblp_500 | football | netscience
##################################################################


if __name__ == '__main__':
    g = nx.read_edgelist("../data/{}/{}_graph.txt".format(database_name, database_name), delimiter=",", nodetype=int)
    get_graph_measure_matrix(g, "karate")

