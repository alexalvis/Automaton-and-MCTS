from decompostAutomata import *
from gridworld import *
import networkx as nx
import numpy as np

class ProductState(object):
    def __init__(self, initgridWorld, initautomaton, initStateG, initStateA):
        self.gridWorld = initgridWorld
        self.automaton = initautomaton
        self.StateG = initStateG
        self.StateA = initStateA
        self.productState = (initStateG, initStateA)

    def transfer(self, productState, jointAction):
        tempStateG = productState[0]
        tempStateA = productState[1]
        newStateG = gridworldTransfer(self.gridWorld, tempStateG, jointAction)
        newStateA = {}
        for key, value in newStateG.items():
            newStateAKey = automata.transfer(tempStateA,key)
            newStateA[newStateAKey] = value
        return newStateA

def gridworldTransfer(g, node, action):
    res = {}
    for dst in g[node]:
        for label in g[node][dst]:
            tempaction = g[node][dst][label]["action"]
            pro = g[node][dst][label]["pro"]
            if tempaction == action:
                res[dst] = pro
    return res

if __name__ == "__main__":
    myobs = []
    gridWorld = gridworld(dim=(11, 11), deterministic=False, r1conn=FOUR_CONNECTED, r2conn=FOUR_CONNECTED, obs=set(myobs))
    automaton = automata()
    StateG = ((0,5),(6,7))
    StateA = "0"
    productState = ProductState(gridWorld, automaton, StateG, StateA)