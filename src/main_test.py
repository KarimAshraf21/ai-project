import networkx as nx
import matplotlib.pyplot as plt
import Methods as m
import AlgoTest as at

#e = [('a', 'c'), ('a', 'e'), ('c', 'b'), ('b', 'e'),('e', 'f'),('e', 'd')]
'''g=nx.DiGraph()'''
#g=nx.MultiDiGraph()
#g.add_edges_from(e)
graph = at.Graph({
    "A": {"B": 10, "C": 3},
    "B": {"C": 1, "D": 2},
    "C": {"B": 4, "D": 8, "E": 2},
    "D": {"E": 7},
    "E": {"D": 9}
     }, {
    'A': 2,
    'B': 3,
    'C': 4,
    'D': 5,
})
#graph.a_star('A','D')
#graph.uniform_cost('S','G')



#print(graph.adjacency_list)
g=nx.MultiGraph(graph.adjacency_list)
nx.draw_networkx(graph, with_labels=True)
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
