import networkx as nx
import matplotlib.pyplot as plt

'''# main input conversion

graph = {'A': {'B': 10, 'C': 3},
         'B': {'C': 1, 'D': 2},
         'C': {'B': 4, 'D': 8, 'E': 2},
         'D': {'E': 7},
         'E': {'D': 9}}

# Convert integer weights to dictionaries with a single 'weight' element
# karim format conversion
gr = {
    from_: {
        to_: {'weight': w}
        for to_, w in to_nodes.items()
    }
    for from_, to_nodes in graph.items()
}

G = nx.from_dict_of_dicts(gr, create_using=nx.DiGraph)
G.edges.data('weight')

# fathy format conversion

AdjList = {}
for node in graph:
    List = []
    temp = list(graph[node].items())
    for i in temp:
        List.append(i)
    AdjList[node] = List

print(f"here{AdjList}")


nx.draw_networkx(G, with_labels=True)

plt.show()

{
    'A': {'B': 10, 'C': 3},
    'B': {'C': 1, 'D': 2},
    'C': {'B': 4, 'D': 8, 'E': 2},
    'D': {'E': 7},
    'E': {'D': 9}
}
'''

# add node, add adjacent node, add weight , any other adjacent nodes?
dict1 = dict()
dict2 = dict()


def input_graph():
    choice = "yes"
    while choice == "yes":
        node = input("enter node name: ")
        dict1[node] = {}

        adjacency_choice = input("is there any adjacent nodes")
        while adjacency_choice == "yes":
            adjacent_node = input("enter adjacent node: ")
            adjacent_node_edge_weight = int(input("enter edge weight: "))
            dict1[node][adjacent_node] = adjacent_node_edge_weight
            adjacency_choice = input("is there any other adjacent nodes?")

        choice = input("do you want to add other nodes?")

    input_heuristics()


def input_heuristics():
    for nodes_keys in dict1.keys():
        dict2[nodes_keys] = int(input(f"enter heuristic for {nodes_keys}: "))




