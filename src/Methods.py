import networkx as nx
from queue import Queue


def BFS(graph, start_node, goal_node):
    visited = []
    fringe = Queue()
    path = []
    current_node = start_node
    found = False

    fringe.put(current_node)
    while not fringe.empty():
        print(f"fringe: {fringe.queue}")
        current_node = fringe.get()
        print(f"current node {current_node}")
        if (current_node == goal_node):
            print("goal is ", current_node)
            visited.append(current_node)
            found = True
            break
        else:
            visited.append(current_node)
            neighbors_iter = graph.neighbors(current_node)
            neighbors = list(neighbors_iter)
            for neighbor in neighbors:
                print(f"neighbor {neighbor}")
                if neighbor in visited:
                    continue
                else:
                    fringe.put(neighbor)
            neighbors.clear()

    if found:
        print("goal found")
    else:
        print('not found')
    print(visited)
