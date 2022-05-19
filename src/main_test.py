import networkx as nx
import matplotlib.pyplot as plt
import Methods as m
import AlgoTest as at

# main input conversion

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
'''G.edges.data('weight')'''

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


