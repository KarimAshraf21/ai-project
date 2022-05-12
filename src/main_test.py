import networkx as nx
import matplotlib.pyplot as plt
import Methods as m
import AlgoTest as at

#e = [('a', 'c'), ('a', 'e'), ('c', 'b'), ('b', 'e'),('e', 'f'),('e', 'd')]
'''g=nx.DiGraph()'''
#g=nx.MultiDiGraph()
#g.add_edges_from(e)

graph= {'A': {'B': 10, 'C': 3},
 'B': {'C': 1, 'D': 2},
 'C': {'B': 4, 'D':8, 'E': 2},
 'D': {'E': 7},
 'E': {'D': 9}}

# Convert integer weights to dictionaries with a single 'weight' element
gr = {
    from_: {
        to_: {'weight': w}
        for to_, w in to_nodes.items()
    }
    for from_, to_nodes in graph.items()
}

G = nx.from_dict_of_dicts(gr, create_using=nx.DiGraph)
G.edges.data('weight')

AdjList ={}
for node in graph:
    List =[]
    temp = list(graph[node].items())
    for i in  temp:
        List.append(i)
    AdjList[node] = List
print(AdjList)

print(graph.adjacency_list)
G = nx.from_dict_of_dicts(graph, create_using=nx.DiGraph)
g=nx.MultiGraph(graph)
nx.draw_networkx(G, with_labels=True)
plt.show()


'''path=m.DFS(g,'a','f')
subax1 = plt.subplot(121)
nx.draw(g, pos=nx.spring_layout(g),with_labels=True)
path_graph=nx.DiGraph()
path_graph.add_nodes_from(path)
path_graph.add_edges_from()
subax2 = plt.subplot(122)
nx.draw(path_graph,with_labels=True)
plt.show()'''
