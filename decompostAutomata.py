import networkx as nx
import preprocess
import numpy as np
class automata(object):
    def __init__(self):
        self.grf = nx.MultiDiGraph()
        self.rankDict = {}
        self.invertedRank = {}
        self.invertedRank[0] = set()
        self.terminal = set()
        self.almostsureWinning = {}
        self.almostsurePolicy = {}
        self.start = None

    def addNode(self, node):
        self.grf.add_node(node)

    def addEdge(self, start, end, target, distance):
        self.grf.add_edge(start, end, target = target, distance = distance)

    def setStart(self, node):
        self.start = node

    def addTerminal(self, end):
        """
        :param end: the terminal of the whole automata
        :return: the first element in rankDict and invertedRank
        """
        self.invertedRank[0].add(end)
        self.rankDict[end] = 0
        self.terminal.add(end)

    def getPre(self, end):
        rank = self.rankDict[end]
        preSet = set()
        predecessor = self.grf.predecessors(end)
        while True:
            try:
                pre = next(predecessor)
                if pre not in self.rankDict:
                    preSet.add(pre)
            except StopIteration:
                break

        for node in preSet:
            self.rankDict[node] = rank + 1

        return preSet
        ## we can use invertedRank to find the node in every rank

    def getRankDict(self):
        rank = 0
        rankSet = self.invertedRank[rank]
        # print(rankSet)
        while True:
            rank += 1
            preSet = set()
            for node in rankSet:
                preSet = preSet.union(self.getPre(node))
            if len(preSet) == 0:
                break
            self.invertedRank[rank] = preSet
            rankSet = preSet

    def getsubAutomata(self, start):
        ##decompose to get 1 single sub-automata
        subgrf = automata()
        subgrf.setStart(start)
        rank = self.rankDict[start]
        subgrf.addNode(start)
        for end in self.grf[start]:
            if self.rankDict[end] == rank -1:
                target = self.grf[start][end][0]["target"]
                distance = self.grf[start][end][0]["distance"]
                subgrf.addEdge(start, end, target, distance)
                subgrf.addTerminal(end)
        subgrf.getRankDict()
        return subgrf

    def transfer(self, start, state):
        for dst in self.grf[start]:
            target = self.grf[start][dst][0]["target"]
            # tempset = self.almostsureWinning[target]
            if state[0] == target and np.linalg.norm(np.array(state[0]) - np.array(state[1]), ord=1) > 1:
                return dst
        return start

    def decomposeAll(self):
        ##decompose to get all possible subautomata
        grfList = []
        for node in self.grf:
            if self.rankDict[node] != 0:
                tempsub = self.getsubAutomata(node)
                grfList.append(tempsub)
        return grfList

def test():
    auto = automata()
    auto.addNode(0)
    auto.addNode(1)
    auto.addNode(2)
    auto.addNode(3)
    auto.setStart(0)
    auto.addTerminal(3)
    auto.addEdge(0,1,(6,7),1)
    auto.addEdge(0,2,(6,3),1)
    auto.addEdge(1,3,(9,1),1)
    auto.addEdge(2,3,(9,9),1)
    auto.getRankDict()
    # print(auto.rankDict)
    # print(auto.grf[0])
    return auto

if __name__ == "__main__":
    ##test
    # auto = automata()
    # auto.addNode("a")
    # auto.addNode("b")
    # auto.addNode("c")
    # auto.addNode("d")
    # auto.addNode("e")
    # auto.addNode("f")
    # auto.addTerminal("f")
    # auto.setStart("a")
    # auto.addEdge("a", "b")
    # auto.addEdge("a", "d")
    # auto.addEdge("b", "c")
    # auto.addEdge("d", "e")
    # auto.addEdge("c", "f")
    # auto.addEdge("e", "f")
    # print("initialize over")
    # auto.getRankDict()
    # print(auto.invertedRank)
    # subgrf = auto.getsubAutomata("a")
    # for node in subgrf:
    #     print (node)
    auto = test()
    # print(auto.rankDict
    autoList = auto.decomposeAll()
    auto_1 = auto.getsubAutomata(0)
    auto_2 = auto.getsubAutomata(1)
    auto_3 = auto.getsubAutomata(2)
    for temp in autoList:
        resTarget, resDist = preprocess.analyse(temp)
        print(resTarget)
        print(resDist)
    print("auto state is:", auto.transfer(0, ((6, 7), (6, 5))))
