# coding=utf-8
import numpy as np

from env_para import EnvPara
from numpy import random as npr


class AdversarialEnvironment:
    def __init__(self):
        """
        Initialization
        """
        "Read the parameters from the json file."

        self.height = EnvPara.height
        self.width = EnvPara.width
        self.rand = EnvPara.ad_rand
        self.gamma = EnvPara.gamma
        self.I = tuple(EnvPara.I_ad)

        self.A = [tuple(x) for x in EnvPara.A]
        self.A_full = EnvPara.A_full
        self.S = []  # the state space
        self.O = [tuple(x) for x in EnvPara.O]
        self.P = {}  # the transition probability

        self.init_mdp()

    def init_mdp(self):
        """
        This function includes some steps to initialize the MDP,
        includes reading parameters from json file,
        adding more states, setting all the probability to zero, setting the rewards,
        Moreover, it will set the transition probability of the MDP.
        :return: NULL
        """
        self.generate_states()
        self.init_probs()
        if not self.check_probs():
            print("ERROR!! The probs is not correct!!!")
            exit(1)
        return

    def get_feasible_actions(self, s):
        feasible_actions = []
        for a in self.A:
            n_s = tuple(map(lambda x, y: x + y, s, a))
        if n_s in self.S and n_s not in self.O:
            feasible_actions.append(a)
        return feasible_actions

    def generate_states(self):
        """
        This function will add all the possible states into the MDP
        :return: NULL
        """
        for x in range(self.width):
            for y in range(self.height):
                self.S.append((x, y))  # add the s into the mdp
        return

    def deter_trans(self, s, a):
        """
        This function return the next s based on deterministic transition without the transition probability.
        :param s: The current s
        :param a: The current a
        :return: The deterministic next s
        """
        assert s in self.S
        assert s not in self.O
        n_s = tuple(map(lambda x, y: x + y, s, a))
        if n_s not in self.S or n_s in self.O:
            n_s = s
        return n_s

    def init_probs(self):
        """
        This function initialize the transition probability between two states for the large scale MDP
        :return: NULL
        """
        for s in self.S:
            for a in self.A:
                for n_s in self.S:
                    self.P[s, a, n_s] = 0  # not matter which it performs, the it never goes out of sink s

        for s in self.S:
            if s in self.O:
                for a in self.A:
                    self.P[s, a, s] = 1.0  # not matter which it performs, the it never goes out of sink s
            else:
                neighbors = self.find_neighbors(s)
                "by doing the following, the prob of being out of the boundary "
                "and hitting the obstacles are added into the correct transition"
                for a in self.A:
                    n_d_s = self.deter_trans(s, a)
                    for n_s in neighbors:
                        if n_s == n_d_s:
                            self.P[s, a, n_d_s] = 1 - (len(neighbors) - 1) * self.rand
                        else:
                            self.P[s, a, n_s] = self.rand
        return

    def find_neighbors(self, s):
        """
        This function finds the neighbours by 4-connectivity for the current s including itself.
        :param s: The current s
        :return: The neighbours of the current states including itself
        """
        assert s in self.S
        neighbors = []
        for a in self.A:
            n_s = self.deter_trans(s, a)
            if n_s in self.S and n_s not in self.O:
                neighbors.append(n_s)
        neighbors = list(set(neighbors))
        return neighbors

    def get_neighbors_probs(self, s, a, neighbors):
        """
        Return a list of the probs of the neighbour states
        :param s: The current state
        :param a: The current action
        :param neighbors: The neighbour states of the current state taking the action a
        :return: a list of the probs of the neighbour states
        """
        probs = []
        for n_s in neighbors:
            if (s, a, n_s) in self.P.keys():
                probs.append(self.P[s, a, n_s])
        return probs

    def sto_trans(self, s, a):
        """
        This function determines a stochastic transition.
        :param s: The current s
        :param a: The optimal a given by the current policy
        :return: The next s
        """
        neighbors = self.find_neighbors(s)
        probs = self.get_neighbors_probs(s, a, neighbors)
        assert abs(sum(probs) - 1) < 1e-5
        n_s = neighbors[npr.choice(range(len(neighbors)), p=probs)]

        assert (np.linalg.norm(np.array(s) - np.array(n_s)) <= 1)
        return n_s

    def check_probs(self):
        """
        This will check the probability initialized correctly
        :return: NULL
        """
        for s in self.S:
            if s not in self.O:
                neighbors = self.find_neighbors(s)
                for a in self.A:
                    probs = self.get_neighbors_probs(s, a, neighbors)
                    assert abs(np.sum(probs) - 1) < 1e-5
        # print(" ")
        # print("| Congrats! You are doing an amazing job of initializing the environment!!!")
        return True
