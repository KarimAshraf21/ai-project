import networkx as nx
import matplotlib.pyplot as plt
import Methods as m
import AlgoTest as at

#e = [('a', 'c'), ('a', 'e'), ('c', 'b'), ('b', 'e'),('e', 'f'),('e', 'd')]
'''g=nx.DiGraph()'''
#g=nx.MultiDiGraph()
#g.add_edges_from(e)

graph= {'A': {'B': {'weight': 10}, 'C': {'weight': 3}},
 'B': {'C': {'weight': 1}, 'D': {'weight': 2}},
 'C': {'B': {'weight': 4}, 'D': {'weight': 8}, 'E': {'weight': 2}},
 'D': {'E': {'weight': 7}},
 'E': {'D': {'weight': 9}}}


#print(graph.adjacency_list)
G = nx.from_dict_of_dicts(graph, create_using=nx.DiGraph)
#g=nx.MultiGraph(graph)
nx.draw_networkx(G, with_labels=True)
plt.show()


'''path=m.DFS(g,'a','f')
subax1 = plt.subplot(121)
nx.draw(g, pos=nx.spring_layout(g),with_labels=True)
path_graph=nx.DiGraph()
path_graph.add_nodes_from(path)
#path_graph.add_edges_from()
subax2 = plt.subplot(122)
nx.draw(path_graph,with_labels=True)
plt.show()'''
