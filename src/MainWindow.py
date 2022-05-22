import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QMainWindow
from PyQt5.uic import loadUi
import Code as tat
import networkx as nx
import matplotlib.pyplot as plt


class WelcomeScreen(QMainWindow):
    def __init__(self):
        super(WelcomeScreen, self).__init__()
        loadUi("WelcomeScreen.ui", self)
        self.AddNodeButton.clicked.connect(self.addNode)
        self.AddEdgeButton.clicked.connect(self.addEdge)
        self.AddHeuristicButton.clicked.connect(self.setHeuristic)
        self.ResetGraphButton.clicked.connect(self.resetGraph)
        self.AddStartNodeButton.clicked.connect(self.addStart)
        self.ResetStartButton.clicked.connect(self.resetStart)
        self.AddGoalsButton.clicked.connect(self.addGoal)
        self.ResetGoalButton.clicked.connect(self.resetGoal)
        self.DrawGraphButton.clicked.connect(self.drawGraph)
        self.DrawPathButton.clicked.connect(self.drawPath)

    def addNode(self):
        tat.g.add_node(self.NodeBox.text())

    def addEdge(self):
        tat.g.add_edge(self.FromBox.text(), self.ToBox.text(), self.WeightBox.text())

    def setHeuristic(self):
        tat.g.add_heuristic(self.CurrentNodeHeuristicBox.text(), self.HeuristicBox.text())

    def resetGraph(self):
        tat.g.reset_graph()

    def addStart(self):
        tat.g.set_start_node(self.StartNodeBox.text())

    def resetStart(self):
        tat.g.reset_start_node()

    def addGoal(self):
        tat.g.append_goal_nodes(self.GoalNodeBox.text())

    def resetGoal(self):
        tat.g.reset_goal_nodes()

    def drawGraph(self):
        tat.g.format1()
        G = nx.from_dict_of_dicts(tat.g.adjFormat1, create_using=nx.MultiDiGraph)
        nx.draw_networkx(G)
        plt.show()

    def drawPath(self):
        x = self.AlgoSelectBox.currentText()
        if x == 'BreadthFirstSearch':
            tat.g.draw_path(tat.g.BreadthFirstSearch())
        elif x == 'DepthFirstSearch':
            tat.g.draw_path(tat.g.DepthFirstSearch())
        elif x== 'UniformCostSearch':
            tat.g.draw_path(tat.g.uniform_cost())
        elif x== 'GreedySearch':
            tat.g.draw_path(tat.g.Greedy())
        elif x== 'AStarSearch':
            tat.g.draw_path(tat.g.a_star())

        elif x== 'IterativeDeepeningSearch':
            tat.g.draw_path(tat.g.IterativeDeepeningSearch())


# main

app = QApplication(sys.argv)
widget = QtWidgets.QStackedWidget()
mainwindow = WelcomeScreen()
mainwindow.setFixedSize(928, 732)
widget.addWidget(mainwindow)
mainwindow.addNode()
widget.show()
sys.exit(app.exec_())
