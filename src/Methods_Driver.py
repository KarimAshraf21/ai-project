import networkx as nx
import matplotlib.pyplot as plt
import Methods as m
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

#nx.draw_networkx(G,pos=nx.spring_layout(G),with_labels=True)
#m.BFS(G,'A','E')

"""drawing the path"""
path_array=m.DFS(G,'A','E')
path_graph=nx.DiGraph()
path_graph.add_nodes_from(path_array)
for i in range(len(path_array)-1):
 path_graph.add_edge(path_array[i],path_array[i+1])

nx.draw_networkx(path_graph,with_labels=True)
plt.show()

#nx.draw_networkx_edge_labels(G,pos)