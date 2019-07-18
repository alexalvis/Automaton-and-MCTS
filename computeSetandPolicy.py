import decompostAutomata
import gridworld
import preprocess
from reachabilityGame import *
from itertools import product
import pickle


# NORTH = lambda st: (st[0] - 1, st[1])
# NORTH.__name__ = 'N'
# SOUTH = lambda st: (st[0] + 1, st[1])
# SOUTH.__name__ = 'S'
# EAST = lambda st: (st[0], st[1] + 1)
# EAST.__name__ = 'E'
# WEST = lambda st: (st[0], st[1] - 1)
# WEST.__name__ = 'W'
# NORTHEAST = lambda st: (st[0] - 1, st[1] + 1)
# NORTHWEST = lambda st: (st[0] - 1, st[1] - 1)
# SOUTHEAST = lambda st: (st[0] + 1, st[1] + 1)
# SOUTHWEST = lambda st: (st[0] + 1, st[1] - 1)
# STAY = lambda st: (st[0],st[1])
# # Connected-ness Definition
# FOUR_CONNECTED = [NORTH, SOUTH, EAST, WEST]                         #: FOUR_CONNECTED = [NORTH, SOUTH, EAST, WEST]
# DIAG_CONNECTED = [NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST]             #:
# EIGHT_CONNECTED = FOUR_CONNECTED + DIAG_CONNECTED

# Turns
TURN_ROBOT = 'robot'
TURN_ENV = 'env'

def getReachSet(target, dist,length, height):
    res = set()
    for i, j in product(range(length), range(height)):
        if (abs(i - target[0]) + abs(j - target[1])) > dist:
            res.add((target, (i, j)))
    return res

def computeSetandPolicy(auto, mygridWorld):
    # auto = decompostAutomata.test()
    # myobs = []
    # length = 11
    # height = 11
    # mygridWorld = gridworld.gridworld(dim=(11, 11), deterministic=False, r1conn=FOUR_CONNECTED, r2conn=FOUR_CONNECTED, obs=set(myobs))
    grf = mygridWorld.concurrentGraph()
    autoList, avoidList = auto.decomposeAll()
    resTarget_auto = {}
    resSet_auto = {}
    resPolicy_auto = {}
    for i in range(len(autoList)):
        subauto = autoList[i]
        avoid = avoidList[i]
        resTarget, resDist = preprocess.analyse(subauto)
        resTargetSet = set()
        for i in range(len(resTarget)):
            tempSet = getReachSet(resTarget[i], resDist, mygridWorld.rows, mygridWorld.cols)
            resTargetSet = resTargetSet.union(tempSet)
            # print(len(tempSet))
        print(len(resTargetSet))
        result, policy = reachability_game_solver(grf, resTargetSet, resDist, avoid, mygridWorld.robotConnectedness, mygridWorld.envConnectedness)
        resSet_auto[subauto.start] = result
        resPolicy_auto[subauto.start] = policy
        resTarget_auto[subauto.start] = resTarget
    ##save results
    # filename = "preCompute/autoalmostSureSet.pkl"
    # picklefile = open(filename, "wb")
    # pickle.dump(resSet_auto, picklefile)
    # picklefile.close()
    #
    # filename = "preCompute/autoalmostSurePolicy.pkl"
    # picklefile = open(filename, "wb")
    # pickle.dump(resPolicy_auto, picklefile)
    # picklefile.close()
    #
    # filename = "preCompute/autotarget.pkl"
    # picklefile = open(filename, "wb")
    # pickle.dump(resTarget_auto, picklefile)
    # picklefile.close()

    return resSet_auto, resPolicy_auto, resTarget_auto

if __name__ =="__main__":
    computeSetandPolicy()