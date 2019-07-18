import decompostAutomata
import gridworld
import pickle
# import inference
import numpy as np
import computeSetandPolicy
import sim_online
from inference import Inference

NORTH = lambda st: (st[0] - 1, st[1])
NORTH.__name__ = 'North'
SOUTH = lambda st: (st[0] + 1, st[1])
SOUTH.__name__ = 'South'
EAST = lambda st: (st[0], st[1] + 1)
EAST.__name__ = 'East'
WEST = lambda st: (st[0], st[1] - 1)
WEST.__name__ = 'West'
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

def oneTimeSimulate(almostSureSet, almostSurePolicy, TargetSet):
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
    # filename = "pre_compute/autoalmostSureSet.pkl"
    # picklefile = open(filename, "wb")
    # pickle.dump(almostSureSet, picklefile)
    # picklefile.close()
    # with open(filename, "rb") as f1:
    #     almostSureSet = pickle.load(f1)

    # filename = "pre_compute/autoalmostSurePolicy.pkl"
    # picklefile = open(filename, "wb")
    # pickle.dump(almostSurePolicy, picklefile)
    # picklefile.close()
    # with open(filename, "rb") as f2:
    #     almostSurePolicy = pickle.load(f2)

    # filename = "pre_compute/autotarget.pkl"
    # picklefile = open(filename, "wb")
    # pickle.dump(TargetSet, picklefile)
    # picklefile.close()
    # with open(filename, "rb") as f3:
    #     TargetSet = pickle.load(f3)

    # input("111")
    # subauto = automaton.getsubAutomata(automaton.start)
    dist = 1
    stateA = initStateA
    stateG = initStateG
    inference_online = Inference()
    traj = []
    traj.append(stateG)
    visitedstate = []
    visitedstate.append(stateA)
    flag = 1
    while (stateA not in automaton.terminal):
        stateG, traj, inference_online = sim_online.simulate(almostSureSet[stateA], almostSurePolicy[stateA], TargetSet[stateA], stateG, traj, inference_online)
        print(traj)
        if checksink(stateG, dist) == False:
            flag = 0
            visitedstate.append(-1)       ##-1 means sink
            break
        tempstateA = automaton.transfer(stateA, stateG)
        if tempstateA != stateA:
            visitedstate.append(tempstateA)
            stateA = tempstateA

    return flag, visitedstate, traj

if __name__ == "__main__":
    # almostSureSet, almostSurePolicy, TargetSet = computeSetandPolicy.computeSetandPolicy(automaton, mygridWorld)
    filename = "pre_compute/autoalmostSureSet.pkl"
    with open(filename, "rb") as f1:
        almostSureSet = pickle.load(f1)

    filename = "pre_compute/autoalmostSurePolicy.pkl"
    with open(filename, "rb") as f2:
        almostSurePolicy = pickle.load(f2)

    filename = "pre_compute/autotarget.pkl"
    with open(filename, "rb") as f3:
        TargetSet = pickle.load(f3)

    catch = 0
    reach = 0
    visitedstateA = []
    trajG = []
    for i in range(10):
        print("Start new Simulation")
        flag, visitedstate, traj = oneTimeSimulate(almostSureSet, almostSurePolicy, TargetSet)
        if flag == 0:
            visitedstateA.append(visitedstate)
            trajG.append(traj)
            catch += 1
        else:
            visitedstateA.append(visitedstate)
            trajG.append(traj)
            reach += 1
        print("traj is:", traj)
        print("visited state is:", visitedstate)
        # input("111")

    print("catch time is:", catch)
    print("finsh time is:", reach)

    filename = "visitedstateA.pkl"
    picklefile = open(filename, "wb")
    pickle.dump(visitedstateA, picklefile)
    picklefile.close()

    filename = "trajG.pkl"
    picklefile = open(filename, "wb")
    pickle.dump(trajG, picklefile)
    picklefile.close()