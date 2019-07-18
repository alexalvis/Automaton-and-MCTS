import networkx as nx
from itertools import product
import pickle
# from deterministic_concurrent_game_v2 import *
# from stochastic_concurrent_game import *

# Default Actions
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

# Turns
TURN_ROBOT = 'robot'
TURN_ENV = 'env'


class DeterministicGridWorld(object):
    """
    Represents a deterministic Grid-World.

    """
    # TODO: DeterministicGridWorld should inherit from Abstract GridWorld class. Define abstraction later. (30 Aug 2017)

    def __init__(self, dim, obs=set(), robotConn=FOUR_CONNECTED, envConn=FOUR_CONNECTED):
        self.rows = dim[0]      #: Number of rows in the gridworld
        self.cols = dim[1]      #: Number of columns in the gridworld
        self.obs = obs          #: Set of static obstacles in the world

        self.robotConnectedness = robotConn
        self.envConnectedness = envConn

        self._turnGraph = None
        self._concurrentGraph = None

    @property
    def turnGraph(self):
        """
        Computes the turn-based graph representing the interaction between robot and
        environment.

        :return: A Labeled Graph with states represented as 3-tuple, :math:`(pos_{r}, pos_{e}, turn)`.
        :rtype: ``networkx.MultiDiGraph`` object.

        .. note:: It is assumed that *environment* plays first.
        """

        """ Assumption: Environment plays first. """
        if self._turnGraph is None:

            # Define graph
            grf = nx.MultiDiGraph()

            # Generate nodes of graph
            for p1, q1, p2, q2 in product(range(self.rows), range(self.cols), repeat=2):
                grf.add_node(n=((p1, q1), (p2, q2), TURN_ROBOT))
                grf.add_node(n=((p1, q1), (p2, q2), TURN_ENV))

            # Generate edges of graph
            for st1, st2, turn in grf.nodes():
                # If state is robot-state: Add edges according to robot's actions
                if turn == TURN_ROBOT:
                    for action in self.robotConnectedness:
                        newSt = action(st1)                                                                                                      ###where is the function action?
                        if 0 <= newSt[0] < self.rows and 0 <= newSt[1] < self.cols:
                            grf.add_edge(u=(st1, st2, turn), v=(newSt, st2, TURN_ENV), action=action)

                elif turn == TURN_ENV:
                    for action in self.envConnectedness:
                        newSt = action(st2)
                        if 0 <= newSt[0] < self.rows and 0 <= newSt[1] < self.cols:
                            grf.add_edge(u=(st1, st2, turn), v=(st1, newSt, TURN_ROBOT), action=action)

            self._turnGraph = grf

        return self._turnGraph

    def concurrentGraph(self):
        """
        Computer the concurrent game graph representing the interaction between robot and environment
        :return: A Labeled Graph with states represented as 3-tuple, : math: 'start_state, later_state, action'
        :rtype； ''networkx.MultiGraph'' object.

        ..note:: The environment and robot play at same time, so we dont care about whose turn
        """
        if self._concurrentGraph is None:
            #Define Graph
            grf = nx.MultiDiGraph()

            #Generate nodes of graph
            for p1, q1,p2, q2 in product(range(self.rows), range(self.cols), repeat = 2):
                grf.add_node(((p1, q1), (p2, q2)))
            for st1, st2 in grf.nodes():            #st1 is robot, st2 is environment
                for action in product(self.robotConnectedness, self.envConnectedness):
                    newSt_r = action[0](st1)
                    newSt_e = action[1](st2)
                    if (0 <= newSt_r[0] < self.rows and 0 <= newSt_r[1] < self.cols and 0 <= newSt_e[0] < self.rows and 0 <= newSt_e[1] < self.cols):
                        grf.add_edge((st1, st2), (newSt_r, newSt_e), action=action)
                        ##in the problem we do not care whose turn it will be since this is a concurrent game

            self._concurrentGraph = grf
        return self._concurrentGraph


class StochasticGridWorld(object):

    """
    Represent a stochastic gridWorld
    """
    def __init__(self, dim, obs=set(), robotConn=FOUR_CONNECTED, envConn=FOUR_CONNECTED):
        self.rows = dim[0]      #: Number of rows in the gridworld
        self.cols = dim[1]      #: Number of columns in the gridworld
        self.obs = obs          #: Set of static obstacles in the world

        self.robotConnectedness = robotConn
        self.envConnectedness = envConn

        self._turnGraph = None
        self._concurrentGraph = None

    def concurrentGraph(self):
        """
        Computer the concurrent game graph representing the interactiion between robot and environment
        :return: a labeled graph states represented as 4-tuple: math: 'start_state, later_state, action, probability'
        :rtype: 'networkx.MultiGraph'
        """

        if self._concurrentGraph is None:
            # Define Graph
            grf = nx.MultiDiGraph()
            for p1,q1,p2,q2 in product(range(self.rows), range(self.cols), repeat = 2):
                grf.add_node(((p1,q1),(p2,q2)))
            for st1, st2 in grf.nodes():
                for action in product(self.robotConnectedness, self.envConnectedness):
                    ##newSt_r = action[0](st1)
                    ##newSt_e = action[1](st2)
                    ##consider every possible action should be legal, so we should first make sure the high pobability answer will remain in the grid world,
                    ##If the newSt is outside the grid world, then we will not consider this action
                    # if (0 <= newSt_r[0] < self.rows and 0 <= newSt_r[1] < self.cols and 0 <= newSt_e[0] < self.rows and 0 <= newSt_e[1] < self.cols):
                    #     dict_r = check_pro(st1,action[0],self.robotConnectedness)    ##dict_r is a dictionary that key is state and value is probability
                    #     dict_e = check_pro(st2,action[1],self.envConnectedness)      ##dict_e is similar to dict_r
                    #     for newSt_r_temp, pro_r in dict_r:              ##newSt_r_temp is the key of dict,the new state and pro_r is the probability reach the newSt_r_temp
                    #         for newSt_e_temp, pro_e in dict_e:          ##similar to the above
                    #             grf.add_edge(u=(st1, st2), v=(newSt_r_temp, newSt_e_temp), action=action, probability = pro_r * pro_e)
                    dict_r = self.check_pro_r(st1, action[0], self.robotConnectedness)
                    dict_e = self.check_pro_e(st2, action[1], self.envConnectedness)
                    for new_st_r, pro_r in dict_r.items():
                        for new_st_e, pro_e in dict_e.items():
                            grf.add_edge((st1, st2), (new_st_r, new_st_e), action = action, pro = pro_r * pro_e)
            self._concurrentGraph = grf
            return self._concurrentGraph


    def check_pro_r(self,state,action,Conn):
        """
        available_dict = {}
        available_count = 0
        for action_temp in Conn:
            if action_temp != action:
                temp_st = action_temp(state)
                if (0 <= temp_st[0] < self.rows and 0 <= temp_st[1] < self.cols):
                    available_count += 1
                    available_dict[temp_st] = 0.1
        actual_st = action(state)
        available_dict[actual_st] = 1 - 0.1*available_count
        return available_dict
        """
        pro_dict = {}
        # pro_dict[state] = 0.0
        temp_st = action(state)
        if (0 <= temp_st[0] < self.rows and 0 <= temp_st[1] < self.cols):
            pro_dict[temp_st] = 1.0
        else:
            pro_dict[state] = 1.0
        # for action_temp in Conn:
        #     if action_temp != action:
        #         temp_st = action_temp(state)
        #         if (0 <= temp_st[0] < self.rows and 0 <= temp_st[1] < self.cols):
        #             pro_dict[temp_st] = 0.05
        #         else:
        #             pro_dict[state]+= 0.05
        return pro_dict

    def check_pro_e(self,state,action,Conn):
        """
        available_dict = {}
        available_count = 0
        for action_temp in Conn:
            if action_temp != action:
                temp_st = action_temp(state)
                if (0 <= temp_st[0] < self.rows and 0 <= temp_st[1] < self.cols):
                    available_count += 1
                    available_dict[temp_st] = 0.1
        actual_st = action(state)
        available_dict[actual_st] = 1 - 0.1*available_count
        return available_dict
        """
        pro_dict = {}
        pro_dict[state] = 0.05
        temp_st = action(state)
        if (0 <= temp_st[0] < self.rows and 0 <= temp_st[1] < self.cols):
            pro_dict[temp_st] = 0.8
        else:
            pro_dict[state] += 0.8
        for action_temp in Conn:
            if action_temp != action:
                temp_st = action_temp(state)
                if (0 <= temp_st[0] < self.rows and 0 <= temp_st[1] < self.cols):
                    pro_dict[temp_st] = 0.05
                else:
                    pro_dict[state]+= 0.05
        # sum = 0.0
        # for keys in pro_dict:
        #     print(pro_dict[keys])
        #     sum+=pro_dict[keys]
        # if (sum!= 1.0):
        #     print("False", sum)
        return pro_dict


def gridworld(dim, r1conn, r2conn, obs=set(), deterministic=True):
    """
    Constructs a gridworld of given dimensions and connectivity for robot and environment.

    :param dim: Dimension of gridworld.
    :type dim: 2-tuple of (xdim, ydim)
    :param r1conn: Connectivity for robot, i.e. the permissible actions for the robot.
    :type r1conn: One amongst the set - {gridworld.FOUR_CONNECTED, gridworld.DIAG_CONNECTED, gridworld.EIGHT_CONNECTED}
    :param r2conn: Connectivity for environment, i.e. the permissible actions for the environment.
    :type r2conn: One amongst the set - {gridworld.FOUR_CONNECTED, gridworld.DIAG_CONNECTED, gridworld.EIGHT_CONNECTED}
    :param obs: Set of obstacles.
    :type obs: Set of 2-tuples, :math:`(x, y)`.
    :param deterministic: Whether world is deterministic or non-deterministic.
    :type deterministic: Boolean
    :rtype: :class:`~DeterministicGridWorld` object.
    """
    if deterministic:
        gridWorld = DeterministicGridWorld(dim=dim, obs=obs, robotConn=r1conn, envConn=r2conn)
        # filename = "./graph/DeterministicGridWorldObj.pkl"
        # file = open(filename, "w")
        # pickle.dump(gridWorld, file)
        # file.close()
        return gridWorld
    else:
        gridWorld = StochasticGridWorld(dim = dim, obs = obs, robotConn=r1conn, envConn=r2conn)
        #filename = "./graph/StochasticGridWorldObj.pkl"
        #file = open(filename, "w")
        #pickle.dump(gridWorld, file)
        #file.close()
        return gridWorld
        #raise NotImplementedError('Non-deterministic grid-worlds are not yet implemented.')

# def filter(grf, r):
#     nodes_to_remove = []
#     for node in grf.nodes():
#         st1 = node[0]
#         st2 = node[1]
#         if ((st1[0] - st2[0])**2 + (st1[1] - st2[1])**2) <= r:
#             nodes_to_remove.append(node)
#     grf.remove_nodes_from(nodes_to_remove)
#     return grf

if __name__ == '__main__':
    #myobs = [(0, 4), (3, 2)]
    myobs = []
    myGrid = gridworld(dim=(11, 11), deterministic=False, r1conn=FOUR_CONNECTED, r2conn=FOUR_CONNECTED, obs=set(myobs))
    grf = myGrid.concurrentGraph()
    #grf = filter(grf , 2)
    W = set()
    W1 = set()
    for i,j in product(range(11),range(11)):          ##（9,5）
        if (abs(i-9) + abs(j-5)) > 1:
            W.add(((9,5),(i,j)))
    # for i,j in product(range(11), range(11)):         ## (6,0)
    #     if (abs(i - 6)+ abs(j - 0)) > 1:
    #         W.add(((6, 0), (i, j)))
    # for i,j in product(range(11), range(11)):         ## (6,10)
    #     if (abs(i - 6) + abs(j - 10)) > 1:
    #         W.add(((6, 10), (i, j)))
    print(len(W))
    result_reach = reachability_game_solver(grf, W, FOUR_CONNECTED, FOUR_CONNECTED)
    filename = "./stochastic/Set_(9,5).pkl"
    pickfile = open(filename, "wb")
    pickle.dump(result_reach, pickfile)
    pickfile.close()
    print("Len of Reachability set is", len(result_reach))
    # filename = "D:\RBE Program\concurrent_omega_regular\Set.pkl"
    # ResSet = set()
    # with open(filename, "rb") as f:
    #     ResSet = pickle.load(f)
    # print(len(ResSet))
    # AllSet = set()
    # for p1, q1, p2, q2 in product(range(11), range(11), repeat=2):
    #     AllSet.add(((p1, q1), (p2, q2)))
    # Set1 = AllSet - result_reach
    # print("Len of Set1 is:" , len(Set1))
    # Set2 = set()
    # Set2 = AllSet - W
    # print("Len of Set2 is:", len(Set2))
    # Set1 = ExchangeSet(Set1)
    # Set2 = ExchangeSet(Set2)
    # print("Len of Set1 is:" , len(Set1))
    # print("Len of Set2 is:" , len(Set2))
    #
    # result1 = safety_game_solver(grf, Set1, FOUR_CONNECTED, FOUR_CONNECTED)
    # print("Compute result1 is over")
    # result2 = safety_game_solver(grf, Set2, FOUR_CONNECTED, FOUR_CONNECTED)
    # print("Len of result1 is:", len(result1))
    # print("Len of result2 is:", len(result2))
    # result1 = ExchangeSet(result1)
    # result2 = ExchangeSet(result2)
    # print("After Exchange: \n")
    # print("Len of result1 is:", len(result1))
    # print("Len of result2 is:", len(result2))
    # filename1 = "./Set1.pkl"
    # pickfile = open(filename1, "wb")
    # pickle.dump(result1, pickfile)
    # pickfile.close()
    # filename2 = "./Set2.pkl"
    # pickfile = open(filename2, "wb")
    # pickle.dump(result2, pickfile)
    # pickfile.close()
    # print('States: ', grf.number_of_nodes(), ' Edges: ', grf.number_of_edges())
