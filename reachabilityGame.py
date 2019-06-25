import decompostAutomata
import gridworld
import productAuto
import pickle

def reachability_game_solver(g, W, dist, robotConn, envConn):
    """
    :param g: graph g
    :param W: the reachability region
    :return: final region
    """
    policy = {}
    W_formal = W
    W_temp, policy = getPre_reachability(g, W, dist, robotConn, envConn, policy)
    W = W.union(W_temp)
    while W != W_formal:
        W_formal = W
        W_temp, policy = getPre_reachability(g, W, dist, robotConn, envConn, policy)
        W = W.union(W_temp)
    return W, policy

def getPre_reachability(g, W, dist, robotConn, envConn, policy):
    """
    :param g: graph g
    :param W: the reachability region
    :param robotConn: robot connecting
    :param envConn:  environment connecting
    :param filename: the file store the node and action
    :return: the predecessors of W
    """
    W_temp = set()
    for node in W:
        predecessor = g.predecessors(node)
        while True:
            try:
                pre = next(predecessor)
                if checkDist(pre,dist) == False or pre in W:
                    continue
                # nodeDict = GetDict_node(g, pre)
                for action_r in robotConn:
                    tempset = set()
                    for action_e in envConn:
                        dst = transfer(g, pre, (action_r,action_e))
                        if dst != None:
                            tempset = tempset.union(dst)
                    if tempset.issubset(W) and len(tempset) != 0 and checkDistSet(tempset, dist) == True:
                        if pre not in policy:
                            policy[pre] = action_r.__name__
                        W_temp.add(pre)
                        break
                    # flag = check_in_W(W, action_r, envConn, nodeDict, pre, node)
                    # if flag == True:
                    #     file.write("node is: " + str(pre) + "  the action can ensure reaching the set in one step is: " + str(action_r) + "\n")
                    #     W_temp.add(pre)
                    #     break
            except StopIteration:
                break;
    print ("W_temp size is", len(W_temp))
    return W_temp, policy

def checkDistSet(set, r):
    for node in set:
        if checkDist(node,r) == False:
            return False
    return True

def checkDist(node, r):
    if (abs(node[0][0]-node[1][0]) + abs(node[0][1] - node[1][1])) <= r:
        return False
    return True

def transfer(g, start, action):
    tempset = set()
    for dst in g[start]:
        for label in g[start][dst]:
            action_temp = g[start][dst][label]["action"]
            if action == action_temp:
                tempset.add(dst)
    return tempset