import os
import networkx as nx
import matplotlib.pyplot as plt
import re
import json

def extract_value_from_file(file_path):
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if "Multi-class correctness" in line:
                return float(line.split(':')[-1].strip())
    return None

def create_tree(directory):
    G = nx.DiGraph()
    # find all subfolders in directory and subfolder name contains the text "set", e.g., "set2_3":
    dirnames = [dirname for dirname in os.listdir(directory) if "set" in dirname]
    # build nodes:
    for dirname in dirnames:
        i = -1
        # extract all filenames in the subfolder:
        dirpath = os.path.join(directory, dirname)
        filenames = os.listdir(dirpath)
        # read info.json:
        info_file = os.path.join(dirpath, "info.json")
        with open(info_file) as f:
            info = json.load(f)
        i = info["depth_index"]
        j = info["breadth_index"]
        parent = info["parent_name"]
        p_i = info["parent_depth_index"]
        p_j = info["parent_breadth_index"]
        # if node does not exist:
        if (i, j) not in G.nodes():
            G.add_node((i, j))
            if parent != "":
                G.add_edge((p_i, p_j), (i, j))
            value = -1
            for filename in filenames:
                if filename.endswith("_stats.txt"):
                    value = extract_value_from_file(os.path.join(dirpath, filename))
            G.nodes[(i, j)]['value'] = value

    return G


def plot_as_graph(G, out_file=None):
    """ Plot the graph as a networkx graph."""
    pos = nx.spring_layout(G)
    values = [G.nodes[node]['value'] for node in G.nodes()]

    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=False, node_size=500, ax=ax, node_color="skyblue")
    # color root node in red:
    root_node = [node for node in G.nodes() if node[0] == 0][0]
    nx.draw_networkx_nodes(G, pos, nodelist=[root_node], node_color="red", node_size=500, ax=ax)
    # color leaf nodes in green:
    # leaf_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
    # nx.draw_networkx_nodes(G, pos, nodelist=leaf_nodes, node_color="green", node_size=500, ax=ax)

    # Annotate nodes with their hierarchy and values
    for node, (x, y) in pos.items():
        depth, level = node#.split("_")
        #label = f"{{{depth}, {level}}}\n{G.nodes[node]['value']:.2f}"
        label = f"set{depth}_{level}\n{G.nodes[node]['value']:.2f}"
        ax.system_text(x, y, s=label, horizontalalignment='center', verticalalignment='center',
                       fontsize=10, color="black")

    if out_file:
        plt.savefig(out_file, format="PNG")
    else:
        plt.show()

    #plt.show()
    if out_file is not None:
        plt.savefig(out_file)
        print(f"Saved tree to {out_file}")
    plt.close()


def plot_as_tree(G, out_file=None):
    """ Plot the graph as a tree."""
    # Use Graphviz to lay out the graph in a tree structure
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB')

    fig, ax = plt.subplots(figsize=(8, 12))  # Adjust the size as needed
    nx.draw(G, pos, with_labels=False, node_size=500, ax=ax, node_color="skyblue", arrows=False)

    # Color the root node in red
    root_node = [node for node in G.nodes() if G.in_degree(node) == 0][0]
    nx.draw_networkx_nodes(G, pos, nodelist=[root_node], node_color="red", node_size=500, ax=ax)

    # Annotate nodes with their hierarchy and values
    for node, (x, y) in pos.items():
        depth_level = node
        value = G.nodes[node]['value']
        label = f"set{depth_level}\n{value:.2f}"
        ax.system_text(x, y - 10, s=label, horizontalalignment='center', verticalalignment='center', fontsize=10, color="black")

    if out_file:
        plt.savefig(out_file, format="PNG")
        print(f"Saved tree to {out_file}")
    else:
        plt.show()

    plt.close()


def plot_as_tree_ver5(G, out_file=None):
    """ Plot the graph as a tree with the root node in red and the longest path in yellow. """
    plt.figure(figsize=(12, 8))  # Initialize the figure

    # Define horizontal spacing for the nodes
    horizontal_spacing = 1.0

    # Create a layout for our nodes based on levels
    levels = {}
    for node in nx.topological_sort(G):
        if len(G.pred[node]) == 0:  # This is a root node
            levels[node] = 0
        else:
            levels[node] = max(levels[parent] for parent in G.pred[node]) + 1

    pos = {}  # Initialize positions dictionary
    for level in range(max(levels.values()) + 1):
        nodes_at_level = [node for node, node_level in levels.items() if node_level == level]
        for i, node in enumerate(nodes_at_level):
            pos[node] = ((i + 0.5) * horizontal_spacing / len(nodes_at_level) - horizontal_spacing / 2, -level)

    # Define node colors
    node_colors = ['red' if len(G.pred[node]) == 0 else 'skyblue' for node in G.nodes()]

    # Identify the longest path
    longest_path = nx.dag_longest_path(G)

    # Draw the graph with node colors based on the node_colors list
    nx.draw(G, pos, with_labels=False, node_size=500, node_color=node_colors, edge_color='gray')

    # Highlight the longest path in yellow
    nx.draw_networkx_edges(G, pos, edgelist=list(nx.utils.pairwise(longest_path)), edge_color='yellow', width=2)

    # Draw node labels with node values
    labels = {node: f"{node}\n{G.nodes[node]['value']:.2f}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    if out_file:  # Save the plot to a file if out_file is specified
        plt.savefig(out_file, bbox_inches='tight')
        print(f"Saved tree plot to {out_file}")
    else:
        plt.show()  # Show the plot if no out_file is specified

    plt.close()  # Close the plot to free up memory

# Example usage:
# G = nx.DiGraph()  # Or however you define your graph
# plot_as_tree_ver5(G, 'my_tree_plot.png')


def create_search_graph_plot(root_directory):
    G = create_tree(root_directory)
    out_file = os.path.join(root_directory, "tree.png")
    #plot_as_graph(G, out_file)
    #plot_as_tree(G, out_file)
    plot_as_tree_ver5(G, out_file)

if __name__ == "__main__":
    root_directory = "/home/guy/code/grounding_tmp3/CUB/threecls/BaseProgramClassifierV2/ViT_L_14_336px/ethan2310/files_programs_CUB_ethan_23_10_23_Friday/programs"
    root_directory = "/home/guy/code/grounding_tmp3/CUB/threecls/BaseProgramClassifierV2/ViT_L_14_336px/files_programs_CUB_ethan_23_10_23/30-10-2023_15-28-59/programs/"
    root_directory = "tmp3/CUB/threecls/BaseProgramClassifierV2/ViT_L_14_336px/files_programs_CUB_ethan_23_10_23/07-11-2023_18-41-20/programs/"
    root_directory = "tmp3/CUB/threecls/BaseProgramClassifierV2/ViT_L_14_336px/files_programs_CUB_ethan_23_10_23/07-11-2023_19-21-03/programs/"
    G = create_tree(root_directory)
    out_file = os.path.join(root_directory, "tree.png")
    #plot_as_tree_ver2(G, out_file)
    plot_as_tree_ver5(G, out_file)
    #plot_as_tree_ver3(G, out_file, level_width=12, vert_gap=0.2, xcenter=0.5)













#
# def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, xcenter=0.5):
#     """If there is a cycle that is reachable from root, then this will see infinite recursion."""
#     if not nx.is_tree(G):
#         raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')
#
#     if root is None:
#         if isinstance(G, nx.DiGraph):
#             root = next(iter(nx.topological_sort(G)))
#         else:
#             root = random.choice(list(G.nodes))
#
#     def _hierarchy_pos(G, root, width=1.0, vert_gap=0.2, xcenter=0.5, pos=None, parent=None, parsed=None):
#         if parsed is None:
#             parsed = []
#         if pos is None:
#             pos = {root: (xcenter, 1)}
#         else:
#             pos[root] = (xcenter, 1 - vert_gap * len(parsed))
#         parsed.append(root)
#         children = list(G.neighbors(root))
#         if not isinstance(G, nx.DiGraph):
#             children = [node for node in children if node not in parsed]
#         if len(children) != 0:
#             dx = width / 2
#             nextx = xcenter - width / 2 - dx / 2
#             for child in children:
#                 nextx += dx
#                 pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
#                                      xcenter=nextx, pos=pos, parent=root, parsed=parsed)
#         return pos
#
#     return _hierarchy_pos(G, root, width, vert_gap, xcenter)
#
# def plot_as_tree_ver2(G, out_file=None):
#     """ Plot the graph as a tree. A version that can work on all computers in our cluster."""
#     root = [n for n, d in G.in_degree() if d == 0]  # Root of a directed tree
#     if len(root) != 1:
#         raise ValueError('The graph does not have a unique root node or is not a tree.')
#     root = root[0]
#     pos = hierarchy_pos(G, root=root)  # get positions in a hierarchy layout
#
#     plt.figure(figsize=(10, 8))
#     nx.draw(G, pos=pos, with_labels=True, node_size=500, node_color="skyblue")
#
#     # Color the root node in red
#     nx.draw_networkx_nodes(G, pos, nodelist=[root], node_color="red", node_size=500)
#
#     # Optionally color the leaf nodes in green
#     # leaf_nodes = [n for n, d in G.out_degree() if d == 0]
#     # nx.draw_networkx_nodes(G, pos, nodelist=leaf_nodes, node_color="green", node_size=500)
#
#     # Annotate nodes with their hierarchy and values
#     for node, (x, y) in pos.items():
#         label = f"{node}\n{G.nodes[node]['value']:.2f}"
#         plt.text(x, y-0.1, s=label, ha='center', va='center', fontsize=8)
#
#     if out_file:
#         plt.savefig(out_file, bbox_inches='tight')
#         print(f"Saved tree to {out_file}")
#     else:
#         plt.show()
#     plt.close()
#
#
# import matplotlib.pyplot as plt
# import networkx as nx
#
#
# def plot_as_tree_ver3(G, out_file=None, level_width=2, vert_gap=0.2, xcenter=0.5):
#     """
#     Plot a tree with the root at the top and leaves at the bottom.
#
#     Parameters:
#     G (networkx.Graph): The graph representing the tree.
#     out_file (str): Path to the output file to save the plot.
#     level_width (float): Horizontal space allocated for each level of the tree.
#     vert_gap (float): Vertical gap between levels of the tree.
#     xcenter (float): Starting x position for the root node.
#     """
#
#     if not nx.is_tree(G):
#         raise ValueError("The graph G must be a tree.")
#
#     def _place(G, v, x, y, dx):
#         pos[v] = (x, y)
#         neighbors = [n for n in G.neighbors(v) if n not in pos]
#         if neighbors:
#             dx = dx / len(neighbors) * level_width
#             nextx = x - dx * (len(neighbors) - 1) / 2
#             for n in neighbors:
#                 _place(G, n, nextx, y - vert_gap, dx)
#                 nextx += dx
#         else:
#             return
#
#     pos = {}
#     root = [n for n, d in G.in_degree() if d == 0]  # find the root of the tree
#     if len(root) != 1:
#         raise ValueError("The tree does not have a unique root.")
#     root = root[0]
#     _place(G, root, xcenter, 1, 1)  # start the root at xcenter and y = 1
#
#     # Now let's plot the graph
#     fig, ax = plt.subplots()
#     nx.draw(G, pos=pos, with_labels=True, labels={node: node for node in G.nodes()},
#             node_size=500, ax=ax, node_color="skyblue")
#
#     # Color the root node in red
#     nx.draw_networkx_nodes(G, pos, nodelist=[root], node_color="red", node_size=500, ax=ax)
#
#     # Annotate nodes with their values if any
#     for node, (x, y) in pos.items():
#         label = f"{node}\n{G.nodes[node].get('value', ''):.2f}"
#         ax.text(x, y, s=label, horizontalalignment='center', verticalalignment='center',
#                 fontsize=10, color="black")
#
#     plt.axis('off')
#     if out_file:
#         plt.savefig(out_file, format="PNG")
#         print(f"Saved tree plot to {out_file}")
#     else:
#         plt.show()
#     plt.close()
#
# import matplotlib.pyplot as plt
# import networkx as nx
#
# def add_nodes(graph, parent, depth, width, pos, x=0, y=0, sibling_gap=0.5):
#     pos[parent] = (x, y)
#     neighbors = list(graph.neighbors(parent))
#     if neighbors:
#         dx = (width / 2 ** depth) * sibling_gap
#         nextx = x - (dx * (len(neighbors) - 1)) / 2
#         for neighbor in neighbors:
#             if neighbor not in pos:
#                 add_nodes(graph, neighbor, depth+1, width, pos, nextx, y-1, sibling_gap)
#                 nextx += dx
#
# def plot_as_tree_ver4(G, out_file=None):
#     pos = {}
#     # Assuming G is a directed graph and we can find a root node as the node with in-degree of 0
#     root_nodes = [n for n, d in G.in_degree() if d == 0]
#     if not root_nodes:
#         raise ValueError("No root node identified")
#     if len(root_nodes) > 1:
#         raise ValueError("Multiple root nodes identified")
#
#     # Increase the sibling_gap parameter to add more space between sibling nodes
#     add_nodes(G, root_nodes[0], 1, len(G.nodes), pos, sibling_gap=2.0)
#
#     plt.figure(figsize=(12, 8))
#     nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue")
#
#     # If values are present as node attributes, we can add them as labels
#     labels = {n: f"{n}\n{G.nodes[n]['value']:.2f}" for n in G.nodes}
#     nx.draw_networkx_labels(G, pos, labels=labels)
#
#     plt.title("Tree Structure (Version 4)")
#     plt.axis('off')  # Turn off the axis
#     if out_file:
#         plt.savefig(out_file, bbox_inches='tight')
#     plt.show()
#
#
# import matplotlib.pyplot as plt
# import networkx as nx
# import numpy as np
