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

    def __init__(self):
        self.path_array = []
        self.adjacency_list = dict()
        self.heuristics_list = dict()
        self.start_node = None
        self.goal_nodes = []
        self.adjFormat1 = dict()
        self.adjFormat2 = dict()
        self.isDirected = True

    def set_type(self, type):
        if type == 'Directed':
            self.isDirected = True

        elif type == 'UnDirected':
            self.isDirected = False

    def add_node(self, node):
        if node == "":
            return
        else:
            self.adjacency_list[node] = {}
        print(self.adjacency_list)

    def add_heuristic(self, node, heuristic):
        self.heuristics_list[node] = int(heuristic)
        print(self.heuristics_list)

    def add_edge(self, node_from, node_to, edge_weight):
        if edge_weight is None:
            edge_weight = 1

        if self.isDirected:
            if node_from in self.adjacency_list.keys():
                self.adjacency_list[node_from][node_to] = int(edge_weight)
            elif node_from not in self.adjacency_list.keys():
                self.add_node(node_from)
                self.add_edge(node_from, node_to, edge_weight)
            if node_to not in self.adjacency_list.keys():
                self.adjacency_list[node_to] = {}
            print(self.adjacency_list)

        elif not self.isDirected:
            if node_from in self.adjacency_list.keys() and node_to in self.adjacency_list.keys():
                self.adjacency_list[node_from][node_to] = int(edge_weight)
                self.adjacency_list[node_to][node_from] = int(edge_weight)
            elif node_from not in self.adjacency_list.keys() and node_to not in self.adjacency_list.keys():
                self.add_node(node_from)
                self.add_node(node_to)
                self.add_edge(node_from, node_to, edge_weight)
            elif node_from in self.adjacency_list.keys() and node_to not in self.adjacency_list.keys():
                self.add_node(node_to)
                self.add_edge(node_from, node_to, edge_weight)

            elif node_from not in self.adjacency_list.keys() and node_to in self.adjacency_list.keys():
                self.add_node(node_from)
                self.add_edge(node_from, node_to, edge_weight)
            print(self.adjacency_list)

    def reset_graph(self):
        self.adjacency_list.clear()
        self.heuristics_list.clear()
        self.path_array.clear()
        self.start_node = None
        self.goal_nodes.clear()
        self.path_array.clear()

        print(self.adjacency_list)
        print(self.heuristics_list)

    def get_neighbors(self, v):
        return self.adjFormat2[v]

    def set_start_node(self, start_node):
        self.start_node = start_node
        print(self.start_node)

    def reset_start_node(self):
        self.start_node = None
        print(self.start_node)

    def append_goal_nodes(self, goal_node):
        self.goal_nodes.append(goal_node)
        print(self.goal_nodes)

    def reset_goal_nodes(self):
        self.goal_nodes.clear()
        print(self.goal_nodes)

    def BreadthFirstSearch(self):
        # some variables
        visited = []
        queue_fringe = Queue()
        path = []
        current_node = self.start_node
        # dictionary that keeps track of parents to find path
        parent = dict()
        parent[self.start_node] = None
        found = False
        # creating the graph
        self.format1()
        G = nx.from_dict_of_dicts(self.adjFormat1, create_using=nx.MultiDiGraph)

        queue_fringe.put(current_node)
        while not queue_fringe.empty():  # iterates until no nodes left to visit
            print(f"fringe: {queue_fringe.queue}")
            # the node that should be expanded is the node first node in the queue fringe
            current_node = queue_fringe.get()
            print(f"current node {current_node}")
            if current_node in self.goal_nodes:
                # print("goal is ", current_node)
                visited.append(current_node)
                found = True
                goal_node = current_node
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

    def DepthFirstSearch(self):
        # some variables
        visited = []
        stack_fringe = []
        path = []
        current_node = self.start_node
        # dictionary that keeps track of parents to find path
        parent = dict()
        parent[self.start_node] = None
        found = False
        # creating the graph
        self.format1()
        G = nx.from_dict_of_dicts(self.adjFormat1, create_using=nx.MultiDiGraph)

        stack_fringe.append(current_node)

        while len(stack_fringe):  # iterates until no nodes left to visit
            # the node that should be expanded is the node on top of the stack
            current_node = stack_fringe.pop()
            if current_node in self.goal_nodes:
                visited.append(current_node)
                found = True
                goal_node = current_node
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

    def IterativeDeepeningSearch(self):
        # convert adj list to adf list of format 2
        self.format2()
        # some variables
        current_node = self.start_node
        depth = 0
        graph_list = {}
        visited = []
        path = []
        parent = dict()
        parent[current_node] = None
        # loop accross all depth iteration by iteration by callin depth limited search algo
        while True:
            print("Looping at depth %i " % (depth))
            # compare nodes returned from depth limited so they match a goal
            result = self.DepthLimitedSearch(current_node, depth, graph_list, visited, path, parent)
            # if a goal reached then break and return path
            if result in self.goal_nodes:
                return path
            # update depth of the next iteration
            depth = depth + 1
            print(graph_list)
            # clear graph adjacency list and visited list to be recreated again in  the next iteration
            graph_list = {}
            visited = []

    def DepthLimitedSearch(self, node, depth, graph_list, visited, path, parent):
        print(node, depth)
        # update graph adjacnency list and visited list for this iteration
        graph_list[node] = []
        if node not in visited:
            visited.append(node)
        if node not in graph_list.items():
            if node in visited and depth != 0:
                graph_list[node] = self.get_neighbors(node)
        # set parent for each node in graph adjacency list
        for j in graph_list.keys():
            for (i, w) in graph_list[j]:
                parent[i] = j
        # if a goal is found then update path list using the parent dictionary and then reverse it to maintain order
        # then return node
        if depth == 0 and node in self.goal_nodes:
            current_node = node
            path.append(current_node)
            while parent[current_node] is not None:
                path.append(parent[current_node])
                current_node = parent[current_node]
            path.reverse()
            print(graph_list)
            print(path)
            return node
        # if goal not found yet and there is more depth to explore
        # then call depthlimited again (new iteration) over childrean -> explore new depth until reaching a goal then return goal node
        elif depth > 0:
            for (i, weight) in self.get_neighbors(node):
                for j in self.goal_nodes:
                    if j == self.DepthLimitedSearch(i, depth - 1, graph_list, visited, path, parent):
                        return j

    def uniform_cost(self):

        # visited_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # expanded_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        visited_list = set([self.start_node])
        expanded_list = set([])

        # graph_dict contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        graph_dict = {}

        graph_dict[self.start_node] = 0

        self.format2()

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[self.start_node] = self.start_node

        while len(visited_list) > 0:
            node = None

            # find a node with the lowest value of f() - evaluation function
            for v in visited_list:
                if node == None or graph_dict[v] < graph_dict[node]:
                    node = v;

            if node == None:
                print('Path does not exist!')
                return None

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if node in self.goal_nodes:
                reconst_path = []

                while parents[node] != node:
                    reconst_path.append(node)
                    node = parents[node]

                reconst_path.append(self.start_node)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbors(node):
                # if the current node isn't in both visited_list and expanded_list
                # add it to visited_list and note node as it's parent
                if m not in visited_list and m not in expanded_list:
                    visited_list.add(m)
                    parents[m] = node
                    graph_dict[m] = graph_dict[node] + weight

                # otherwise, check if it's quicker to first visit node, then m
                # and if it is, update parent data and graph_dict data
                # and if the node was in the expanded_list, move it to visited_list
                else:
                    if graph_dict[m] > graph_dict[node] + weight:
                        graph_dict[m] = graph_dict[node] + weight
                        parents[m] = node

                        if m in expanded_list:
                            expanded_list.remove(m)
                            visited_list.add(m)

            # remove node from the visited_list, and add it to expanded_list
            # because all of his neighbors were inspected
            visited_list.remove(node)
            expanded_list.add(node)

        print('Path does not exist!')
        return None

    def a_star(self):

        # visited_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # expanded_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        visited_list = set([self.start_node])
        expanded_list = set([])
        self.format2()
        # graph_dict contain
        # s current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        graph_dict = {}

        graph_dict[self.start_node] = 0

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[self.start_node] = self.start_node

        while len(visited_list) > 0:
            node = None

            # find a node with the lowest value of f() - evaluation function
            for v in visited_list:
                if node == None or graph_dict[v] + self.heuristics_list[v] < graph_dict[node] + self.heuristics_list[
                    node]:
                    node = v;

            if node == None:
                print('Path does not exist!')
                return None

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if node in self.goal_nodes:
                reconst_path = []

                while parents[node] != node:
                    reconst_path.append(node)
                    node = parents[node]

                reconst_path.append(self.start_node)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbors(node):
                # if the current node isn't in both visited_list and expanded_list
                # add it to visited_list and note node as it's parent
                if m not in visited_list and m not in expanded_list:
                    visited_list.add(m)
                    parents[m] = node
                    graph_dict[m] = graph_dict[node] + weight

                # otherwise, check if it's quicker to first visit node, then m
                # and if it is, update parent data and graph_dict data
                # and if the node was in the expanded_list, move it to visited_list
                else:
                    if graph_dict[m] > graph_dict[node] + weight:
                        graph_dict[m] = graph_dict[node] + weight
                        parents[m] = node

                        if m in expanded_list:
                            expanded_list.remove(m)
                            visited_list.add(m)

            # remove node from the visited_list, and add it to expanded_list
            # because all of his neighbors were inspected
            visited_list.remove(node)
            expanded_list.add(node)

        print('Path does not exist!')
        return None

    def Greedy(self):
        self.format2()
        # visited_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # expanded_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        visited_list = set([self.start_node])
        expanded_list = set([])

        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        # g = {}

        # g[start_node] = 0

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[self.start_node] = self.start_node

        while len(visited_list) > 0:
            node = None

            # find a node with the lowest value of f() - evaluation function
            for v in visited_list:
                if node == None or self.heuristics_list[v] < self.heuristics_list[node]:
                    node = v;

            if node == None:
                print('Path does not exist!')
                return None

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if node in self.goal_nodes:
                reconst_path = []

                while parents[node] != node:
                    reconst_path.append(node)
                    node = parents[node]

                reconst_path.append(self.start_node)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbors(node):
                # if the current node isn't in both visited_list and expanded_list
                # add it to visited_list and note node as it's parent
                if m not in visited_list and m not in expanded_list:
                    visited_list.add(m)
                    parents[m] = node
                    # g[m] = g[node] + weight

                # otherwise, check if it's quicker to first visit node, then m
                # and if it is, update parent data and g data
                # and if the node was in the expanded_list, move it to visited_list
                else:
                    if self.heuristics_list[m] > self.heuristics_list[node]:
                        # g[m] = g[node] + weight
                        parents[m] = node

                        if m in expanded_list:
                            expanded_list.remove(m)
                            visited_list.add(m)

            # remove node from the visited_list, and add it to expanded_list
            # because all of his neighbors were inspected
            visited_list.remove(node)
            expanded_list.add(node)

        print('Path does not exist!')
        return None

    def format1(self):
        self.adjFormat1 = {
            from_: {
                to_: {'weight': w}
                for to_, w in to_nodes.items()
            }
            for from_, to_nodes in self.adjacency_list.items()
        }

    def format2(self):
        self.adjFormat2 = {}
        for node in self.adjacency_list:
            List = []
            temp = list(self.adjacency_list[node].items())
            for i in temp:
                List.append(i)
            self.adjFormat2[node] = List

    def draw_path(self, path):  # function takes search function as a parameter
        if path is None:
            return
        else:
            self.path_array = path
            path_graph = nx.DiGraph()
            path_graph.add_nodes_from(self.path_array)
            for i in range(len(self.path_array) - 1):
                path_graph.add_edge(self.path_array[i], self.path_array[i + 1])

            nx.draw_networkx(path_graph, with_labels=True)
            plt.show()


g = Graph()
"""
dict3 = {
    'A': {'B': 10, 'C': 3},
    'B': {'C': 1, 'D': 2},
    'C': {'B': 4, 'D': 8, 'E': 2},
    'D': {'E': 7},
    'E': {'D': 9}
}

dict4 = {
    'A': 2,
    'B': 3,
    'C': 4,
    'D': 5,
    'E': 0
}
graph = Graph(dict3, dict4)
graph.add_node('z')
print(graph.adjacency_list)
graph.add_edge('z','y',3)
print(graph.adjacency_list)
graph.add_heuristic('z',100)
print(graph.heuristics_list)
graph.reset_graph()
print(graph.adjacency_list)
print(graph.heuristics_list)

"""
