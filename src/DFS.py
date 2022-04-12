import networkx as nx
from queue import Queue

graph = {
    'a': ['c'],
    'b': ['d'],
    'c': ['e'],
    'd': ['a', 'd'],
    'e': ['b', 'c']
}

def DepthFirst(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    for node in graph[start]:
        if node not in path:
            newpath = DepthFirst(graph, node, end, path)
            if newpath:
                return newpath

print(DepthFirst(graph, 'd', 'c'))