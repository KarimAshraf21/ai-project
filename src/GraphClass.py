from collections import deque
from queue import Queue

import networkx as nx


class Graph:
    # example of adjacency list (or rather map)
    # adjacency_list = {
    # 'A': [('B', 1), ('C', 3), ('D', 7)],
    # 'B': [('D', 5)],
    # 'C': [('D', 12)]
    # }

    def _init_(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    # heuristic function with equal values for all nodes
    def h(self, n):
        H = {
            'A': 1,
            'B': 1,
            'C': 1,
            'D': 1
        }

        return H[n]

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
        self.uninformed_format()
        graph = nx.from_dict_of_dicts(self.gr, create_using=nx.MultiGraphDiGraph)

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

    def a_star_algorithm(self, start_node, stop_node):
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

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function
            for v in open_list:
                if n == None or g[v] + self.h(v) < g[n] + self.h(n):
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

    def uninformed_format(self):
        self.gr = {
            from_: {
                to_: {'weight': w}
                for to_, w in to_nodes.items()
            }
            for from_, to_nodes in self.adjacency_list.items()
        }


