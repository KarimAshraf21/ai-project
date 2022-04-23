import networkx as nx
from queue import Queue
from queue import PriorityQueue


def BFS(graph, start_node, goal_node):
    # some variables
    visited = []
    queue_fringe = Queue()
    path = []
    current_node = start_node
    # dictionary that keeps track of parents to find path
    parent = dict()
    parent[start_node] = None
    found = False

    queue_fringe.put(current_node)
    while not queue_fringe.empty():  # iterates until no nodes left to visit
        print(f"fringe: {queue_fringe.queue}")
        # the node that should be expanded is the node first node in the queue fringe
        current_node = queue_fringe.get()
        print(f"current node {current_node}")
        if current_node == goal_node:
            # print("goal is ", current_node)
            visited.append(current_node)
            found = True
            break
        else:
            # getting neighbors and adding them to the visited list
            visited.append(current_node)
            neighbors_iter = graph.neighbors(current_node)
            neighbors = list(neighbors_iter)
            print(neighbors)
            print(parent)

            # assigning parents to nodes
            for neighbor in neighbors:
                if neighbor in parent.keys():
                    list(parent[neighbor]).extend(current_node)
                else:
                    parent[neighbor] = current_node

            # putting unvisited nodes in the fringe
            for neighbor in neighbors:
                print(f"neighbor {neighbor}")
                if neighbor in visited:
                    continue
                else:
                    queue_fringe.put(neighbor)
            neighbors.clear()

    if found:
        print("goal found")
        '''path(goal_node)'''
        # backtracking for getting the parents which are the path
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


def DFS(graph, start_node, goal_node):
    # some variables
    visited = []
    stack_fringe = []
    path = []
    current_node = start_node
    # dictionary that keeps track of parents to find path
    parent = dict()
    parent[start_node] = None
    found = False

    stack_fringe.append(current_node)

    while len(stack_fringe):  # iterates until no nodes left to visit
        # the node that should be expanded is the node on top of the stack
        current_node = stack_fringe.pop()
        if current_node == goal_node:
            visited.append(current_node)
            found = True
            break
        else:
            # getting neighbors and adding them to the visited list
            visited.append(current_node)
            print(f"visited {visited}")
            neighbors_iter = graph.neighbors(current_node)
            neighbors = list(neighbors_iter)
            neighbors.sort()
            neighbors.reverse()
            print(neighbors)
            print(parent)

            # assigning parents to nodes
            for neighbor in neighbors:
                if neighbor in parent.keys():
                    list(parent[neighbor]).extend(current_node)
                else:
                    parent[neighbor] = current_node
            # pushing nodes into the fringe
            for neighbor in neighbors:
                print(f"neighbor {neighbor}")
                if neighbor in visited:
                    continue
                else:
                    stack_fringe.append(neighbor)
            neighbors.clear()

    if found:
        print("goal found")
        '''path(goal_node)'''
        # backtracking for getting the parents which are the path
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


def UCS(graph,start_node,goal_node):
    #lessa hakteb

