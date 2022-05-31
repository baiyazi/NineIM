"""
@Time: 2020年11月10日 10:22:10
@version: 1.0,
@author: WangZhibin,
@description: 社区划分，返回{0: [], 1: []}这种形式的结果

1. Blondel, V.D. et al. Fast unfolding of communities in
    large networks. J. Stat. Mech 10008, 1-12(2008).
"""


import community as community_louvain

def communityPartition(nx_G):
    # nx_G = nx.karate_club_graph()
    partition = community_louvain.best_partition(nx_G)
    # partition字典，按照其计算的社区来统计下，构成形式：{0: [], 1: []}
    community_dict = dict()
    for community_identify in list(set(partition.values())):
        community_dict[community_identify] = [val for val in partition.keys() if partition[val] == community_identify]
    return community_dict


import networkx as nx

g = nx.karate_club_graph()
communityPartition(g)


# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
# import networkx as nx
# # draw the graph
# pos = nx.spring_layout(G)
# # color the nodes according to their partition
# cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
# nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
#                        cmap=cmap, node_color=list(partition.values()))
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# plt.show()
