from collections import deque
from queue import Queue

import networkx as nx
from matplotlib import pyplot as plt


class Graph:
    # example of adjacency list (or rather map)
    # adjacency_list = {
    # 'A': [('B', 1), ('C', 3), ('D', 7)],
    # 'B': [('D', 5)],
    # 'C': [('D', 12)]
    # }

    def __init__(self, adjacency_list, heuristics_list):
        self.adjacency_list = adjacency_list
        self.heuristics_list = heuristics_list

    def get_neighbors(self, v):
        return self.AdjList[v]

    def BreadthFirstSearch(self, start_node, goal_node):
        # some variables
        visited = []
        queue_fringe = Queue()
        path = []
        current_node = start_node
        # dictionary that keeps track of parents to find path
        parent = dict()
        parent[start_node] = None
        found = False
        # creating the graph
        self.format1()
        G = nx.from_dict_of_dicts(self.gr, create_using=nx.MultiDiGraph)

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
                neighbors_iter = G.neighbors(current_node)
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

    def DepthFirstSearch(self, start_node, goal_node):
        # some variables
        visited = []
        stack_fringe = []
        path = []
        current_node = start_node
        # dictionary that keeps track of parents to find path
        parent = dict()
        parent[start_node] = None
        found = False
        # creating the graph
        self.format1()
        G = nx.from_dict_of_dicts(self.gr, create_using=nx.MultiDiGraph)

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
                neighbors_iter = G.neighbors(current_node)
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

    def uniform_cost(self, start_node, stop_node):

        # open_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # closed_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        open_list = set([start_node])
        closed_list = set([])

        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        g = {}

        g[start_node] = 0

        self.format2()

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function
            for v in open_list:
                if n == None or g[v] < g[n]:
                    n = v;

            if n == None:
                print('Path does not exist!')
                return None

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if n == stop_node:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbors(n):
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight

                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return None

    def a_star(self, start_node, stop_node):

        # open_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # closed_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        open_list = set([start_node])
        closed_list = set([])
        self.format2()
        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        g = {}

        g[start_node] = 0

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function
            for v in open_list:
                if n == None or g[v] + self.heuristics_list[v] < g[n] + self.heuristics_list[n]:
                    n = v;

            if n == None:
                print('Path does not exist!')
                return None

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if n == stop_node:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbors(n):
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight

                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return None

    def Greedy(self, start_node, stop_node):
        self.format2()
        # open_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # closed_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        open_list = set([start_node])
        closed_list = set([])

        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        # g = {}

        # g[start_node] = 0

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function
            for v in open_list:
                if n == None or self.heuristics_list[v] < self.heuristics_list[n]:
                    n = v;

            if n == None:
                print('Path does not exist!')
                return None

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if n == stop_node:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbors(n):
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    # g[m] = g[n] + weight

                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                else:
                    if self.heuristics_list[m] > self.heuristics_list[n]:
                        # g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return None

    def format1(self):
        self.gr = {
            from_: {
                to_: {'weight': w}
                for to_, w in to_nodes.items()
            }
            for from_, to_nodes in self.adjacency_list.items()
        }

    def format2(self):
        self.AdjList = {}
        for node in self.adjacency_list:
            List = []
            temp = list(self.adjacency_list[node].items())
            for i in temp:
                List.append(i)
            self.AdjList[node] = List

    def draw_path(self, path):  # function takes search function as a parameter
        self.path_array = path
        path_graph = nx.DiGraph()
        path_graph.add_nodes_from(self.path_array)
        for i in range(len(self.path_array) - 1):
            path_graph.add_edge(self.path_array[i], self.path_array[i + 1])

        nx.draw_networkx(path_graph, with_labels=True)
        plt.show()


graph = Graph({
    'A': {'B': 10, 'C': 3},
    'B': {'C': 1, 'D': 2},
    'C': {'B': 4, 'D': 8, 'E': 2},
    'D': {'E': 7},
    'E': {'D': 9}
}, {
    'A': 2,
    'B': 3,
    'C': 4,
    'D': 5,
    'E': 0
})

'''graph.BreadthFirstSearch('A', 'D')
graph.draw_path(graph.DepthFirstSearch('A', 'E'))'''

graph.draw_path(graph.Greedy('A', 'E'))
