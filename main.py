# encoding=utf-8

import warnings
from setting import config
from NINE.Nine import getFinalEmbedding
from NineIM.NineIM import influence_measure

warnings.filterwarnings("ignore")


################################################################################
database_name = "football"  # 可选数据库：dblp_500 | football | CA-GrQc | netscience
################################################################################

database_template = "data/{}/{}_graph.txt"  # graph的位置


if __name__ == "__main__":
    args = config.parse_args()
    embeddings_ = getFinalEmbedding(database_name, args)

    beta = 0.07
    gama = 0.136
    influence_measure(database_name,beta=beta, gama=gama, embeddings_=embeddings_)
