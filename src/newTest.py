import networkx as nx
import matplotlib.pyplot as plt

'''graph = {
    "A": {"B": 10, "C": 3},
    "B": {"C": 1, "D": 2},
    "C": {"B": 4, "D": 8, "E": 2},
    "D": {"E": 7},
    "E": {"D": 9}
}

gr = {
    from_: {
        to_: {'weight': w}
        for to_, w in to_nodes.items()
    }
    for from_, to_nodes in graph.items()
}
G = nx.from_dict_of_dicts(gr, create_using=nx.DiGraph)
G.edges.data('weight')
'''



graph= {'A': {'B': {'weight': 10}, 'C': {'weight': 3}},
 'B': {'C': {'weight': 1}, 'D': {'weight': 2}},
 'C': {'B': {'weight': 4}, 'D': {'weight': 8}, 'E': {'weight': 2}},
 'D': {'E': {'weight': 7}},
 'E': {'D': {'weight': 9}}}

G=nx.DiGraph(graph)

nx.draw_networkx(G,pos=nx.spring_layout(G),with_labels=True)

#nx.draw_networkx_edge_labels(G,pos)
plt.show()
