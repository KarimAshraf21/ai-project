import networkx as nx
import matplotlib.pyplot as plt
import Methods as m

e = [('a', 'c'), ('a', 'e'), ('c', 'b'), ('b', 'e'),('e', 'f'),('e', 'd')]
'''g=nx.DiGraph()'''
g=nx.MultiDiGraph()
g.add_edges_from(e)



print(g.adj)

path=m.DFS(g,'a','f')
subax1 = plt.subplot(121)
nx.draw(g, pos=nx.spring_layout(g),with_labels=True)
path_graph=nx.DiGraph()
path_graph.add_nodes_from(path)
subax2 = plt.subplot(122)
nx.draw(path_graph,with_labels=True)
plt.show()
