import networkx as nx
from queue import Queue

def BFS(graph, start_node, goal_node):
    visited = []
    fringe = Queue()
    path = []
    current_node = start_node
    parent = dict()
    parent[start_node] = None

    found = False

    fringe.put(current_node)
    while not fringe.empty():  # iterates until no nodes left to visit
        print(f"fringe: {fringe.queue}")
        current_node = fringe.get()
        print(f"current node {current_node}")
        if (current_node == goal_node):
            # print("goal is ", current_node)
            visited.append(current_node)
            found = True
            break
        else:
            visited.append(current_node)
            neighbors_iter = graph.neighbors(current_node)
            neighbors = list(neighbors_iter)
            for neighbor in neighbors:    #problem with appending value
                parent[neighbor]=(current_node)
            for neighbor in neighbors:
                print(f"neighbor {neighbor}")
                if neighbor in visited:
                    continue
                else:
                    fringe.put(neighbor)
            neighbors.clear()

    if found:
        print("goal found")
        '''path(goal_node)'''
        path.append(goal_node)
        while parent[current_node] is not None:
            path.append(parent[current_node])
            current_node = parent[current_node]
        path.reverse()
    else:
        print('not found')
    print(f"visited is {visited}")
    print(parent)
    print(f"path is {path}")
    return path
