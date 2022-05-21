import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QMainWindow
from PyQt5.uic import loadUi
import src.AlgoTest as at
import networkx as nx
import matplotlib.pyplot as plt


def resetGraph():
    at.reset_graph()


class WelcomeScreen(QMainWindow):
    def __init__(self):
        super(WelcomeScreen, self).__init__()
        loadUi("WelcomeScreen.ui", self)
        self.AddNodeButton.clicked.connect(self.addNode)
        self.AddEdgeButton.clicked.connect(self.addEdge)
        self.AddHeuristicButton.clicked.connect(self.setHeuristic)
        self.DrawGraphButton.clicked.connect(at.create_graph())
        self.ResetGraphButton.clicked.connect(at.resetGraph)

    def addNode(self):
        at.add_node(self.NodeBox.text())

    def addEdge(self):
        at.add_edge(self.FromBox.text(), self.ToBox.text(), self.WeightBox.text())

    def setHeuristic(self):
        at.add_heuristic(self.CurrentNodeHeuristicBox.text(), self.HeuristicBox.text())

    def createAndDrawGraph(self):
        g = at.create_graph()
        nx.draw_networkx(g)
        plt.show()


# main

app = QApplication(sys.argv)
widget = QtWidgets.QStackedWidget()
mainwindow = WelcomeScreen()
mainwindow.setFixedSize(928, 732)
widget.addWidget(mainwindow)
mainwindow.addNode()
widget.show()
sys.exit(app.exec_())
