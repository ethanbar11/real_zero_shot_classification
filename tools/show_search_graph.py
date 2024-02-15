import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ovod.utils.vis_optimization import create_search_graph_plot

if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--root_directory", type=str, required=True, help="root directory of the search")
    args = parser.parse_args()
    create_search_graph_plot(args.root_directory)