import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NINE(Node Influence Based Embedding) program.")
    parser.add_argument('--weighted', type=bool, default=False, help='Is the input graph weighted?')
    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')
    parser.add_argument('--beta', type=float, default=1, help='Random selecte from node community parameter. Defalut is 1.')
    parser.add_argument('--global-random-parameter', type=float, default=0.3, help='The global random parameter. Default is 0.3.')
    parser.add_argument('--local-random-parameter', type=float, default=0.2, help='The local random parameter. Default is 0.2.')
    parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')
    parser.add_argument('--window-size', type=int, default=5, help='Length of train window size. Default is 5.')
    return parser.parse_args()