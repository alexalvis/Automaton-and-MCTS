import decompostAutomata
import gridworld
import pickle
# import inference
import numpy as np
import computeSetandPolicy
import sim_onlineV3

NORTH = lambda st: (st[0] - 1, st[1])
NORTH.__name__ = 'N'
SOUTH = lambda st: (st[0] + 1, st[1])
SOUTH.__name__ = 'S'
EAST = lambda st: (st[0], st[1] + 1)
EAST.__name__ = 'E'
WEST = lambda st: (st[0], st[1] - 1)
WEST.__name__ = 'W'
NORTHEAST = lambda st: (st[0] - 1, st[1] + 1)
NORTHWEST = lambda st: (st[0] - 1, st[1] - 1)
SOUTHEAST = lambda st: (st[0] + 1, st[1] + 1)
SOUTHWEST = lambda st: (st[0] + 1, st[1] - 1)
STAY = lambda st: (st[0],st[1])
# Connected-ness Definition
FOUR_CONNECTED = [NORTH, SOUTH, EAST, WEST]                         #: FOUR_CONNECTED = [NORTH, SOUTH, EAST, WEST]
DIAG_CONNECTED = [NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST]             #:
EIGHT_CONNECTED = FOUR_CONNECTED + DIAG_CONNECTED

def decideAction(state, Policy):
    action = np.random.choice(list(Policy[state].keys()), list(Policy[state].values()))
    return action

def stateTransfer(state, action, SystemSto):
    tempstate = np.random.choice(list(SystemSto[state][action].keys()), list(SystemSto[state][action].values()))
    return tempstate

def checksink(state, dist):
    if (abs(state[0][0] - state[1][0]) + abs(state[0][1] - state[1][1])) > dist:
        return True
    return False

def oneTimeSimulate():
    #initialize automaton
    automaton = decompostAutomata.test()
    #initialize gridWorld
    myobs = []
    mygridWorld = gridworld.gridworld(dim=(11, 11), deterministic=False, r1conn=FOUR_CONNECTED, r2conn=FOUR_CONNECTED, obs=set(myobs))

    #set initial state
    initStateG = ((0, 5),(10, 5))
    initStateA = automaton.start
    #get Set and Policy

    # almostSureSet, almostSurePolicy, TargetSet = computeSetandPolicy.computeSetandPolicy(automaton, mygridWorld)
    filename = "pre_compute/autoalmostSureSet.pkl"
    # picklefile = open(filename, "wb")
    # pickle.dump(almostSureSet, picklefile)
    # picklefile.close()
    with open(filename, "rb") as f1:
        almostSureSet = pickle.load(f1)

    filename = "pre_compute/autoalmostSurePolicy.pkl"
    # picklefile = open(filename, "wb")
    # pickle.dump(almostSurePolicy, picklefile)
    # picklefile.close()
    with open(filename, "rb") as f2:
        almostSurePolicy = pickle.load(f2)

    filename = "pre_compute/autotarget.pkl"
    # picklefile = open(filename, "wb")
    # pickle.dump(TargetSet, picklefile)
    # picklefile.close()
    with open(filename, "rb") as f3:
        TargetSet = pickle.load(f3)

    # subauto = automaton.getsubAutomata(automaton.start)
    dist = 1
    stateA = initStateA
    stateG = initStateG
    visitedtarget = []
    flag = 1
    while (stateA not in automaton.terminal):
        stateG = sim_onlineV3.simulate(almostSureSet[stateA], almostSurePolicy[stateA], TargetSet[stateA], stateG)
        if checksink(stateG, dist) == False:
            flag = 0
            break
        visitedtarget.append(stateG[0])
        stateA = automaton.transfer(stateA, stateG)
    # if stateA not in automaton.terminal:
    #     print("The system goes into sink state")
    # else:
    #     print("Mission Complete")
    return flag, visitedtarget

if __name__ == "__main__":
    catch = 0
    reach = 0
    traj = []
    for i in range(100):
        print("Start new Simulation")
        flag, visited = oneTimeSimulate()
        if flag == 0:
            catch += 1
        else:
            traj.append(visited)
            reach += 1

    print("catch time is:", catch)
    print("finsh time is:", reach)