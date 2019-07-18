import math
import pickle
from copy import deepcopy as dcp
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from bayesian_inference.BayesianParam import BAYESIAN_INFERENCE
from env_para import EnvPara


def KL(P, Q):
    epsilon = 1e-9
    P = np.asarray(P) + epsilon
    Q = np.asarray(Q) + epsilon

    divergence = np.sum(P * np.log(P / Q))
    return divergence


class BayesianLearner:
    def __init__(self):
        self.Height = BAYESIAN_INFERENCE.HEIGHT
        self.Width = BAYESIAN_INFERENCE.WIDTH
        self.goals = BAYESIAN_INFERENCE.GOALS
        self.size = [self.Height, self.Width]
        self.A = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}
        self.A_full = {"North": (-1, 0), "South": (1, 0), "West": (0, -1), "East": (0, 1)}
        self.S = self.set_S()  # use list or set? should be in the form of (st1, st2)
        self.obstacles = BAYESIAN_INFERENCE.OBSTACLES  # should be a 2-D state, like (1,3) represent only agent 1
        self.R = self.get_R()  # range
        self.epsilon1 = BAYESIAN_INFERENCE.TRANSITION_PROB1
        self.epsilon2 = BAYESIAN_INFERENCE.TRANSITION_PROB2  # stochastic probability
        self.gamma = BAYESIAN_INFERENCE.GAMMA
        self.threshold = 0.000001
        self.decoys = {}
        self.memory = (EnvPara.I_contr, EnvPara.I_ad)
        self.true_goal = []
        for i in range(len(BAYESIAN_INFERENCE.TERMINAL)):
            self.true_goal.append((BAYESIAN_INFERENCE.TERMINAL[i][0], BAYESIAN_INFERENCE.TERMINAL[i][1]))
        self.distancethreshold = BAYESIAN_INFERENCE.DISTANCETHRESHOLD
        self.goals_reward_value = 100
        self.catch = self.get_catch()
        self.catch_reward = BAYESIAN_INFERENCE.CATCHREWARD

        g = 'g'
        for i in range(len(self.goals)):
            key = g + str(i + 1)
            self.decoys[key] = []
            g_i = tuple([self.goals[i][0], self.goals[i][1]])  ##agent2 认为的goal
            # g_ctrl = tuple([self.goals[i][0], self.goals[i][1]])  ##agent1 自己的goal
            self.decoys[key].append(g_i)

        self.init_decoys = dcp(self.decoys)

        self.P_g = {}
        for key in self.decoys:
            self.P_g[key] = 1 / len(self.decoys.keys())  ##initial belief of agent 2

        filename_P_s1_a_s2 = "pre_compute/filename_P_s1_a_s2.pkl"
        filename_Probability_P1 = "pre_compute/filename_Probability_P1.pkl"
        filename_Pi1_g = "pre_compute/filename_Pi1_g.pkl"
        filename_P1_s_g = "pre_compute/filename_P1_s_g.pkl"
        filename_Probability_P2 = "pre_compute/filename_Probability_P2.pkl"
        filename_reward_g_s_a2 = "pre_compute/filename_reward_g_s_a2.pkl"
        filename_Pi2_g = "pre_compute/filename_Pi2_g.pkl"
        filename_P2_s_g = "pre_compute/filename_P2_s_g.pkl"
        filename_Q = "pre_compute/filename_Q.pkl"

        self.initial_belief = dcp(self.P_g)
        self.P1_s_g = dict.fromkeys(self.P_g)
        self.Pi1_g = dict.fromkeys(self.P_g)
        self.P2_s_g = dict.fromkeys(self.P_g)
        self.Pi2_g = dict.fromkeys(self.P_g)
        self.reward_g_s_a2 = dict.fromkeys(self.P_g)
        self.Q = dict.fromkeys(self.P_g)
        self.Probability_P2 = {}

        # self.P_s1_a_s2 = self.get_P()  ##from state1 take action a transform to state2, this is from system
        # picklefile = open(filename_P_s1_a_s2, "wb")
        # pickle.dump(self.P_s1_a_s2, picklefile)
        # picklefile.close()
        #
        # self.initialPolicy_P2 = self.get_initialPolicy_P2()
        #
        # self.Probability_P1 = self.get_Probability_P1()
        # picklefile = open(filename_Probability_P1, "wb")
        # pickle.dump(self.Probability_P1, picklefile)
        # picklefile.close()
        #
        # initialize the belief and the policy
        # the Policy of P1
        # for decoy in self.Pi1_g.keys():
        #     self.Pi1_g[decoy] = self.get_Policy_P1(decoy,self.Probability_P1)
        #     self.P1_s_g[decoy] = self.get_Transition(decoy, self.Pi1_g[decoy], self.Probability_P1)
        #
        # picklefile = open(filename_Pi1_g, "wb")
        # pickle.dump(self.Pi1_g, picklefile)
        # picklefile.close()

        # picklefile = open(filename_P1_s_g, "wb")
        # pickle.dump(self.P1_s_g, picklefile)
        # picklefile.close()

        # for decoy in self.decoys.keys():
        #     self.Probability_P2[decoy] = self.get_Probability_P2(decoy)
        #     self.reward_g_s_a2[decoy] = self.reward(decoy, self.Probability_P2[decoy])

        # picklefile = open(filename_Probability_P2, "wb")
        # pickle.dump(self.Probability_P2, picklefile)
        # picklefile.close()

        # picklefile = open(filename_reward_g_s_a2, "wb")
        # pickle.dump(self.reward_g_s_a2, picklefile)
        # picklefile.close()

        # for decoy in self.Pi2_g.keys():
        #     self.Pi2_g[decoy], self.Q[decoy] = self.get_Policy_P2(decoy, self.Probability_P2[decoy], self.reward_g_s_a2[decoy])
        #     self.P2_s_g[decoy] = self.get_Transition(decoy, self.Pi2_g[decoy], self.Probability_P2[decoy])

        # picklefile = open(filename_Pi2_g, "wb")
        # pickle.dump(self.Pi2_g, picklefile)
        # picklefile.close()

        # picklefile = open(filename_P2_s_g, "wb")
        # pickle.dump(self.P2_s_g, picklefile)
        # picklefile.close()

        # picklefile = open(filename_Q, "wb")
        # pickle.dump(self.Q, picklefile)
        # picklefile.close()
        #
        # read file directly
        with open(filename_P_s1_a_s2, "rb") as f1:
            self.P_s1_a_s2 = pickle.load(f1)

        with open(filename_Probability_P1, "rb") as f2:
            self.Probability_P1 = pickle.load(f2)

        with open(filename_Pi1_g, "rb") as f3:
            self.Pi1_g = pickle.load(f3)

        with open(filename_P1_s_g, "rb") as f4:
            self.P1_s_g = pickle.load(f4)

        with open(filename_Probability_P2, "rb") as f5:
            self.Probability_P2 = pickle.load(f5)

        with open(filename_reward_g_s_a2, "rb") as f6:
            self.reward_g_s_a2 = pickle.load(f6)

        with open(filename_Pi2_g, "rb") as f7:
            self.Pi2_g = pickle.load(f7)

        with open(filename_P2_s_g, "rb") as f8:
            self.P2_s_g = pickle.load(f8)

        with open(filename_Q, "rb") as f9:
            self.Q = pickle.load(f9)

        self.P_h_g = dcp(self.P_g)  ##agent2 belief
        self.ctrl_belief_P1 = dcp(self.P_h_g)  ##agent1 belief

        for goals in self.decoys:
            if self.decoys[goals][0] in self.true_goal:
                self.ctrl_belief_P1[goals] = 1.0
            else:
                self.ctrl_belief_P1[goals] = 0.0

        self.init_ctrl_belief = dcp(self.ctrl_belief_P1)  ##the initial belief of P1

        self.record = []
        self.true_record = []
        self.total_KL = 0
        self.max_KL = KL([1, 0, 0, 0], [0, 1, 0, 0])
        # print("Bayesian initialize end")

    ##reduce the decoy   ##2019.7.6 no more need in this version
    def reduce(self, decoy):
        self.P_h_g.pop(decoy)
        self.decoys.pop(decoy)
        self.ctrl_belief_P1.pop(decoy)

        temp_sum = sum(self.P_h_g.values())
        flag = False
        for goal in self.P_h_g.keys():
            if self.P_h_g[goal] < 0.01:
                flag = True
                break
        normalize_flag = True
        normalized_p = {}

        ##do the normalization
        if flag:
            for goal in self.decoys:
                self.P_h_g[goal] = 1.0 / len(self.P_h_g)
        else:
            for goal in self.decoys:
                normalized_p[goal] = self.P_h_g[goal] / temp_sum
                if normalized_p[goal] > 0.99:
                    new_temp_sum = temp_sum - self.P_h_g[goal]
                    self.P_h_g[goal] = 0.99
                    for g in self.P_h_g:
                        if not g == goal:
                            self.P_h_g[g] = (1 - 0.99) * self.P_h_g[g] / new_temp_sum
                    normalize_flag = False
                    break
            if normalize_flag:
                self.P_h_g = dcp(normalized_p)

    ##reset the policy and belief
    def reset(self):
        self.P_h_g = dcp(self.P_g)
        self.ctrl_belief_P1 = dcp(self.init_ctrl_belief)
        self.decoys = dcp(self.init_decoys)
        for goals in self.decoys:
            if self.decoys[goals][0] in self.true_goal:
                self.ctrl_belief_P1[goals] = 1.0
            else:
                self.ctrl_belief_P1[goals] = 0.0

        self.record = []
        self.true_record = []
        self.total_KL = 0

    ##get the state set S
    def set_S(self):
        height = self.Height
        width = self.Width
        inner = []
        for p1, q1, p2, q2 in product(range(height), range(width), repeat=2):
            inner.append(((p1, q1), (p2, q2)))
        # print("total number of states: ", len(inner))
        return inner

    ##get the range the discrete agent should stay in
    def get_R(self):
        R = []
        height = self.Height
        width = self.Width
        for p1, q1 in product(range(height), range(width)):
            R.append((p1, q1))
        return R

    def get_catch(self):
        catch = []
        for state in self.S:
            s = tuple(state)
            if (abs(s[0][0] -s[1][0]) + abs(s[0][1] - s[1][1])) <= self.distancethreshold:
                catch.append(s)
        return catch

    ##get the transition probability of each single agent
    def trans_P(self, state, id):
        A = self.A
        R = self.R
        P = {}
        P[state] = {}
        st1 = state[0]
        st2 = state[1]
        if id == 1:
            if st1 not in self.obstacles and st1 not in self.true_goal and tuple(state) not in self.catch:
                epsilon = self.epsilon1
                explore = []  ##single state
                for action in A:
                    temp = tuple(np.array(st1) + np.array(A[action]))
                    explore.append(temp)
                for action in A:
                    unit1 = epsilon / 4
                    P[state][action] = {}
                    P[state][action][st1] = unit1
                    temp_st1 = tuple(np.array(st1) + np.array(A[action]))
                    if temp_st1 in R and list(temp_st1) not in self.obstacles:
                        # print("temp_st1 is in R:" ,temp_st1)
                        P[state][action][temp_st1] = 1 - epsilon  ## difference is here, large probability to move
                        for _st_ in explore:
                            if _st_ != temp_st1:
                                if _st_ in R:
                                    P[state][action][_st_] = unit1
                                else:
                                    P[state][action][st1] += unit1
                    else:  ##next step will out of range or enter obstacles
                        P[state][action][st1] = 1 - epsilon + unit1  ##difference is here, large probability to remain
                        for _st_ in explore:
                            if _st_ != temp_st1:
                                if _st_ in R:
                                    P[state][action][_st_] = unit1
                                else:
                                    P[state][action][st1] += unit1

            else:  ##agent1 now in obstacle or true goal or been caught, so each action will remain in the obstacle
                for action in A:
                    P[state][action] = {}
                    P[state][action][st1] = 1.0
        else:  ## for agent 2, we dont care about obstacles or goals
            if tuple(state) not in self.catch and st1 not in self.obstacles and st1 not in self.true_goal:
                epsilon = self.epsilon2
                explore = []
                for action in A:
                    temp = tuple(np.array(st2) + np.array(A[action]))
                    explore.append(temp)
                for action in A:
                    unit2 = epsilon / 4
                    P[state][action] = {}
                    P[state][action][st2] = unit2
                    temp_st2 = tuple(np.array(st2) + np.array(A[action]))
                    if temp_st2 in R:
                        P[state][action][temp_st2] = 1 - epsilon
                        for _st_ in explore:
                            if _st_ != temp_st2:
                                if _st_ in R:
                                    P[state][action][_st_] = unit2
                                else:
                                    P[state][action][st2] += unit2
                    else:
                        P[state][action][st2] = 1 - epsilon + unit2
                        for _st_ in explore:
                            if _st_ != temp_st2:
                                if _st_ in R:
                                    P[state][action][_st_] = unit2
                                else:
                                    P[state][action][st2] += unit2
            else:
                for action in A:
                    P[state][action] = {}
                    P[state][action][st2] = 1.0

        return P  ##here P is in the form of P[(st1,st2)][a][st_]

    ##get the joint state transfer probability
    def get_P(self):
        P = {}
        S = self.S
        A = self.A
        R = self.R
        ##epsilon1 = self.epsilon1
        ##epsilon2 = self.epsilon2
        for state in S:  ##state:(st1, st2)
            Pro1 = self.trans_P(state, 1)  ## transfer probability dict
            Pro2 = self.trans_P(state, 2)  ## trnasfer probability dict
            P[state] = {}
            for a1, a2 in product(A, A):
                P[state][(a1, a2)] = {}
                for st1_ in Pro1[state][a1].keys():
                    for st2_ in Pro2[state][a2].keys():
                        P[state][(a1, a2)][(st1_, st2_)] = Pro1[state][a1][st1_] * Pro2[state][a2][st2_]
        return P  ##here P is in the form of P[(st1,st2)][(a1,a2)][(st1_, st2_)]

    ##get the initial policy of P2, read from file
    def get_initialPolicy_P2(self):
        # with open(filename, 'rb') as input:
        #     Pi2 = pickle.load(input)
        Pi2 = {}
        for state in self.S:
            s = tuple(state)
            if s not in Pi2:
                Pi2[s] = {}
            for a in self.A:
                Pi2[s][a] = 0.25

        return Pi2

    ##this is reward function, reference in draft
    def reward(self, goals, P):
        alfa = 1.0
        reward_g_s_a2 = {}
        S = self.S
        A = self.A
        obstacles = self.obstacles
        for state in S:
            s = tuple(state)
            if s not in reward_g_s_a2:
                reward_g_s_a2[s] = {}
            if s[0] not in self.decoys[goals] and s[0] not in obstacles:
                for a in A:
                    reward_g_s_a2[s][a] = 0.0
                    for s_ in P[s][a].keys():
                        if s_ in self.catch:
                            reward_g_s_a2[s][a] += alfa * P[s][a][s_] * self.catch_reward  ##catch probability multiply catch reward
            else:
                for a in A:
                    reward_g_s_a2[s][a] = -self.goals_reward_value

        return reward_g_s_a2  ## should be in the form of reward[state][a2]

    ##get_transfer probability of P1
    def get_Probability_P1(self):
        P = self.P_s1_a_s2
        counter_Probability = self.initialPolicy_P2
        control_Probability = {}
        S = self.S
        A = self.A
        goals = self.true_goal
        obstacles = self.obstacles
        for state in S:
            s = tuple(state)
            if s[0] not in goals and s[0] not in obstacles and s not in self.catch:
                control_Probability[state] = {}
                for a_control in A:
                    control_Probability[state][a_control] = {}
                    for a_counter in A:
                        for s_ in P[state][(a_control, a_counter)].keys():
                            if s_ not in control_Probability[state][a_control].keys():
                                control_Probability[state][a_control][s_] = 0
                            control_Probability[state][a_control][s_] += P[state][(a_control, a_counter)][s_] * \
                                                                         counter_Probability[state][a_counter]
            else:
                control_Probability[state] = {}
                for a_control in A:
                    if a_control not in control_Probability[state].keys():
                        control_Probability[state][a_control] = {}
                    control_Probability[state][a_control][state] = 1.0
        return control_Probability  ##should be in the form of P[state][action][state_], the state should be joint state, the action should be a single action, state_ should be
        # the joint state

    ##get the transition probability of P2 based on the policy of P1[goal]
    def get_Probability_P2(self, goal):
        P = self.P_s1_a_s2
        counter_Probability = self.Pi1_g[goal]
        control_Probability = {}
        S = self.S
        A = self.A
        # goals = self.true_goal
        obstacles = self.obstacles
        for state in S:
            control_Probability[state] = {}
            s = tuple(state)
            if s[0] not in self.decoys[goal] and s[0] not in obstacles and s not in self.catch:
                for a_control in A:
                    control_Probability[state][a_control] = {}
                    for a_counter in A:
                        for s_ in P[state][(a_counter, a_control)].keys():
                            if s_ not in control_Probability[state][a_control].keys():
                                control_Probability[state][a_control][s_] = 0
                            control_Probability[state][a_control][s_] += P[state][(a_counter, a_control)][s_] * \
                                                                         counter_Probability[state][a_counter]
            else:
                for a_control in A:
                    if a_control not in control_Probability[state].keys():
                        control_Probability[state][a_control] = {}
                    control_Probability[state][a_control][state] = 1.0
        return control_Probability  ##the return probability is in the form of P[state][action][state_]

    ##get the policy of P1, the optimal strategy is SSP
    def get_Policy_P1(self, goals, P, init=None):
        ##transform Dict to vector
        def Dict2Vec(values, states):
            v = []
            for s in states:
                v.append(values[tuple(s)])
            return np.array(v)

        ## the sigma function used in MDP
        def Sigma(s, a, P, V_):
            # print("s is:", s)
            total = 0.0
            for s_ in P[s][a].keys():
                # print("s_ is:" ,s_)
                if s_ != s:
                    total += P[s][a][s_] * V_[s_]
            return total

        def init_V(S, goals, g, ng=None):
            V, V_ = {}, {}
            for state in S:
                s = tuple(state)
                if s not in V:
                    V[s], V_[s] = 0.0, 0.0
                if s[0] in self.decoys[goals]:
                    V[s], V_[s] = g, g
                    # print("s is:" ,s)
                if ng != None and s[0] in ng:
                    V[s], V_[s] = 0.0, 0.0
                if s in self.catch:
                    V[s], V_[s] = -self.catch_reward, -self.catch_reward
            return V, V_

        Policy_P1 = {}
        Q = {}
        V_record = []
        S = self.S
        A = self.A
        # P = self.Probability_P1
        threshold = self.threshold
        gamma = self.gamma
        ng = self.obstacles
        tau = 1
        if init == None:
            V, V_ = init_V(S, goals, self.goals_reward_value, ng)
        else:
            V, V_ = dcp(init), dcp(init)
            for unsafe in V:
                s = tuple(unsafe)
                if s[0] in ng or s in self.catch:
                    V[unsafe], V_[unsafe] = 0, 0
        V_current = Dict2Vec(V, S)
        V_last = Dict2Vec(V_, S)
        it_count = 1
        while it_count == 1 or np.inner(V_current - V_last, V_current - V_last) > threshold:
            V_record.append(sum(V.values()))
            for s in S:
                V_[tuple(s)] = V[tuple(s)]
                # if tuple(s)[0] == self.decoys[goals][0]:
                #     print("11111111111111111:   ",V[tuple(s)], "s is:   " ,s)
            for state in S:
                s = tuple(state)
                if s[0] not in self.decoys[goals] and s[0] not in ng and s not in self.catch:
                    if s not in Policy_P1:
                        Policy_P1[s] = {}
                    if s not in Q:
                        Q[s] = {}

                    for a1 in A:
                        core = gamma * Sigma(s, a1, P, V_)
                        Q[s][a1] = np.exp(core)  ## will this lead any problem?
                    Q_s = sum(Q[s].values())
                    # if s[0] == (self.decoys[goals][0][0] -1, self.decoys[goals][0][1]):
                    #     print("Q_s on goal neighbour is:" ,Q_s, "state is:", s)
                    for a1 in A:
                        if a1 in P[s].keys():
                            Policy_P1[s][a1] = Q[s][a1] / Q_s
                    V[s] = tau * np.log(Q_s)
                else:
                    if s not in Policy_P1:
                        Policy_P1[s] = {}
                        for a1 in A:
                            Policy_P1[s][a1] = 0.00001
            V_current, V_last = Dict2Vec(V, S), Dict2Vec(V_, S)
            print("it_count number is:", it_count)
            it_count += 1
            # for state in S:
            #     s = tuple(state)
            #     if s[0] == self.decoys[goals][0]:
            #         print("value at (6,0) is:",V[s])
        return Policy_P1  # should be in the form of P[state][action] state should be the joint state and action is the single action, this is based on the aim goal

    # get the policy of P2, using MDP, the optimal strategy is maximize the total accumulated reward
    def get_Policy_P2(self, goals, P, reward, init=None):
        # transform dict to vector
        print("enter get_Policy_P2")

        def Dict2Vec(values, states):
            v = []
            for s in states:
                v.append(values[tuple(s)])
            return np.array(v)

        # in normal form MDP, the initial V value should all be 0
        def init_V(S, goals, g=100.0, ng=None):
            V, V_ = {}, {}
            for state in S:
                s = tuple(state)
                if s not in V:
                    V[s], V_[s] = 0.0, 0.0
                if s[0] in self.decoys[goals]:
                    V[s], V_[s] = -g, -g
                    # if ng != None and s in ng:
                    #     V[s], V_[s] = 0.0, 0.0
                if s in self.catch:
                    V[s], V_[s] = self.catch_reward, self.catch_reward
            return V, V_

        ## Q-function
        def Q_func(s, a, P, V, reward, gamma):
            Q = reward[s][a]
            for s_ in P[s][a].keys():
                Q += gamma * P[s][a][s_] * V[s_]
            return Q

        Policy_P2 = {}
        Q = {}
        V_record = []
        S = self.S
        A = self.A
        # P = self.Probability_P1
        threshold = self.threshold
        gamma = self.gamma
        ng = self.obstacles

        tau = 1

        if init == None:
            V, V_ = init_V(S, goals, 100.0, ng)
        else:
            V, V_ = dcp(init), dcp(init)
            for unsafe in V:
                s = tuple(unsafe)
                if s[0] in ng:
                    V[unsafe], V_[unsafe] = 0.0, 0.0
        V_current = V
        V_last = V_
        it_count = 1
        while it_count == 1 or np.inner(V_current - V_last, V_current - V_last) > threshold:
            V_record.append(sum(V.values()))
            for s in S:
                V_[tuple(s)] = V[tuple(s)]
            for state in S:
                s = tuple(state)
                if s[0] not in self.decoys[goals] and s[0] not in ng:
                    if s not in Policy_P2:
                        Policy_P2[s] = {}
                    if s not in Q:
                        Q[s] = {}
                    for a in A:
                        Q[s][a] = Q_func(s, a, P, V_, reward, gamma)
                    summation = 0.0
                    for a in A:
                        core = np.exp(Q[s][a] / tau)
                        summation += core
                    V[s] = tau * np.log(summation)
                    policy_sum = 0.0
                    for a in A:  # calculate the miu_b(a|s)
                        Policy_P2[s][a] = np.exp((Q[s][a] - V[s]) / tau)
                        policy_sum += Policy_P2[s][a]
                    for a in A:  # normalize
                        Policy_P2[s][a] = Policy_P2[s][a] / policy_sum

                else:
                    if s not in Policy_P2:
                        Policy_P2[s] = {}
                    if s not in Q:
                        Q[s] = {}
                    for a in A:
                        Q[s][a] = Q_func(s, a, P, V_, reward, gamma)
                    if s[0] in self.decoys[goals]:
                        V[s] = -self.goals_reward_value  ## reach the goal
                        for a in A:
                            Policy_P2[s][a] = 0.25  ##need discussion
                    else:
                        V[s] = 0  ##sink state
                        for a in A:
                            Policy_P2[s][a] = 0.25  ##need discussion
            V_current, V_last = Dict2Vec(V, S), Dict2Vec(V_, S)
            print("it_count number:", it_count)
            it_count += 1
        return Policy_P2, Q

    ##get the transition probability
    def get_Transition(self, goals, Pi, P):
        T = {}
        # Pi = self.Pi1_g[goals]
        S = self.S
        # P = self.P_s1_a_s2

        for state in S:
            s = tuple(state)
            T[s] = {}
            if s[0] not in self.decoys[goals] and s[0] not in self.obstacles and s not in self.catch:
                for a in Pi[s].keys():
                    for s_ in P[s][a].keys():
                        if s_ not in T[s]:
                            T[s][s_] = 0.0
                        T[s][s_] += P[s][a][s_] * Pi[s][a]
            else:
                T[s][s] = 1.0
        return T  # in the form of T[state][state_], both are joint state

    # def newP(self, last_s, s, a2):
    #     Pro = {}
    #     action1 = np.array(s[0]) - np.array(last_s[0])
    #     action1 = tuple(action1)
    #     for action in self.A:
    #         if self.A[action] == action1:
    #             a1 = action
    #             break
    #     for goals in self.decoys:
    #         Pro[goals] = self.P_s1_a_s2[last_s][(a1, a2)][s] * self.Pi1_g[goals][last_s][a1]
    #
    #     return Pro


    # the bayesian inference
    def Bayesian_inference(self, traj, record_flag=False, display_flag=False):
        self.total_KL = 0
        last_s = traj[0]
        flag = False
        # the state in traj should be joint state
        for s in traj:
            # jump over the first one
            if not flag:
                flag = True
                continue

            # if the current state in the decoy, then reduce the decoy
            # for goal in self.decoys:
            #     if s[0] in self.decoys[goal]:
            #         self.reduce(goal)

            if display_flag:
                print('------------------')
                print('|current state:', s)
                for goals in self.P_h_g:
                    print('|belief at ', goals, ':', self.P_h_g[goals])
                print('------------------')

            # find the max value of the transition probability
            action1 = np.array(s[0]) - np.array(last_s[0])
            action1 = tuple(action1)
            if action1 == (0, 0):
                for action in self.A.keys():
                    temp = tuple(np.array(last_s[0]) + np.array(self.A[action]))
                    if temp not in self.R:
                        act1 = action
            else:
                for action in self.A.keys():
                    if self.A[action] == action1:
                        act1 = action
                        break

            max_value = -1
            for goals in self.decoys:
                try:
                    if self.Pi1_g[goals][last_s][act1] > max_value:
                        max_value = self.Pi1_g[goals][last_s][act1]
                except KeyError:
                    continue

            if max_value <= 0.001:
                "Rest the value"
                self.P_h_g = dcp(self.P_g)
                last_s = s
                self.total_KL += KL(list(self.P_h_g.values()), list(self.ctrl_belief_P1.values())) / self.max_KL
            else:
                for goals in self.decoys:
                    try:
                        self.P_h_g[goals] = self.P_h_g[goals] * self.Pi1_g[goals][last_s][act1]
                    except KeyError:
                        self.P_h_g[goals] = 0.0

                temp_sum = sum(self.P_h_g.values())
                for goals in self.decoys:
                    self.P_h_g[goals] /= temp_sum

                last_s = s
                self.total_KL += KL(list(self.P_h_g.values()), list(self.ctrl_belief_P1.values())) / self.max_KL

        if record_flag:
            self.record.append(dcp(self.P_h_g))


    def semi_optimal_strategy(self, s):
        belief = self.P_h_g
        Q = self.Q
        miu_b_P2 = {}
        tau = 1.0
        for a in self.A:
            if a not in miu_b_P2.keys():
                miu_b_P2[a] = 0.0
            for goals in self.decoys:
                miu_b_P2[a] += belief[goals] * Q[goals][s][a] / tau
            miu_b_P2[a] = np.exp(miu_b_P2[a])

            # normalize
        temp_sum = sum(miu_b_P2.values())
        for a in self.A:
            miu_b_P2[a] /= temp_sum

        return miu_b_P2

    # the one-shot Bayesian Inference
    def Bayesian_inference_one_shot(self, s, ad_s=None, record_flag=True, display_flag=False):
        last_s = self.memory

        for goal in self.decoys:
            if s[0] in self.decoys[goal] and s[0] not in self.true_goal:
                self.reduce(goal)
                break

        if ad_s is not None:
            for goal in self.decoys:
                if ad_s[0] in self.decoys[goal] and ad_s[0] not in self.true_goal:
                    self.reduce(goal)
                    print("reduced!!!!", "\ngoal is:", goal, "\ns is:", s, "\nad_s is:", ad_s)

        for goals in self.decoys:
            try:
                self.P_h_g[goals] = self.P_h_g[goals] * self.P1_s_g[goals][last_s][s]

            except Exception as e:
                print(e)
                self.P_h_g[goals] = 0.0

                # if self.P1_s_g[goals][last_s][s] > 1:
                #     print("In bayesian Inference part existing P1_s_g larger than 1")

        temp_sum = sum(self.P_h_g.values())
        if abs(temp_sum) < 1e-30:
            print("Error, the summation of P_h_g is 0!!")
            print("last_s:", last_s, "  s:", s)
            print("P1_s_g:", self.P1_s_g[goals][last_s][s])
            print(temp_sum)

        flag = True
        normalized_p = {}
        try:
            for goal in self.decoys:
                normalized_p[goal] = self.P_h_g[goal] / temp_sum

                if normalized_p[goal] > 0.99:
                    new_temp_sum = temp_sum - self.P_h_g[goal]
                    self.P_h_g[goal] = 0.99

                    for g in self.decoys:
                        if not g == goal:
                            self.P_h_g[g] = (1 - 0.99) * self.P_h_g[g] / new_temp_sum
                        if math.isinf(self.P_h_g[g]):
                            self.P_h_g[g] = 0.005
                    flag = False
                    break

            if flag:
                self.P_h_g = dcp(normalized_p)
        except Exception as e:
            print(e)
        if abs(sum(self.P_h_g.values())) > 1.1:
            print("Error, the summation of P_h_g is not 1!!!")
            print("last_s:", last_s, "  s:", s)
            for goal in self.decoys:
                print(self.P_h_g[goal])
            print(self.P1_s_g[goals][last_s][s])
        self.memory = s

        if record_flag:
            self.record.append(dcp(self.P_h_g))
            self.total_KL += KL(list(self.P_h_g.values()), list(self.ctrl_belief_P1.values())) / self.max_KL
            temp_KL = KL(list(self.P_h_g.values()), list(self.ctrl_belief_P1.values()))
            # if math.isnan(temp_KL):
            #     print("Error! temp_KL is nan")
        if display_flag:
            print('------------------')
            print('|current state:', s)
            for goals in self.P_h_g:
                print('|belief at ', goals, ':', self.P_h_g[goals])
            print('current controlled belief:', self.ctrl_belief_P1)
            print('------------------')
        # probability = self.semi_optimal_strategy(s)
        # print("--------------------------Action----------------")
        # for a in self.A:
        #     print(a, "    ", probability[a])
        # print("——————————————————————————")


if __name__ == '__main__':
    print("enter main")
    learner = BayesianLearner()
    input("111")
    # start_time = time.time()
    # learner.reset()
    # end_time = time.time()
    # interval = end_time - start_time
    # print(interval)
    filename3 = "joint_traj.pkl"
    with open(filename3, "rb") as f3:
        traj = pickle.load(f3)
    print(learner.decoys['g1'])
    print(learner.decoys['g2'])
    print(learner.decoys['g3'])
    print(traj)
    belief_1 = []
    belief_2 = []
    belief_3 = []
    for element in traj[1:]:
        if not tuple(element) == learner.memory:
            learner.Bayesian_inference_one_shot(tuple(element), display_flag=True)
            try:
                belief_1.append(learner.P_h_g['g1'])
            except KeyError as e:
                print("g1 has been removed")
            try:
                belief_2.append(learner.P_h_g['g2'])
            except KeyError as e:
                print("g2 has been removed")
            try:
                belief_3.append(learner.P_h_g['g3'])
            except KeyError as e:
                print("g3 has been removed")

    # probability = learner.semi_optimal_strategy(tuple(element))
    # for a in learner.A:
    #     print(probability[a])
    # print(learner.total_KL)
    # print(KL([1, 0.0, 0.0], [0.0, 1, 0.0]))
    # print(KL([.5, 0, .5], [0, 1, 0]))
    # print(KL([0, 0, 1], [0, 1, 0]))
    # print(KL([0, 1], [1, 0]))
    # print(KL([1, 0], [0, 1]))

    fig, ax1 = plt.subplots()
    plt.plot(range(len(belief_1)), belief_1, label='g1', linewidth=2.0, linestyle='--')
    plt.plot(range(len(belief_2)), belief_2, label='g2', linewidth=2.0, linestyle='-')
    plt.plot(range(len(belief_3)), belief_3, label='g3', linewidth=2.0, linestyle='-.')
    # plt.rcParams.update({'font.size': 22})
    plt.tick_params(labelsize=10)
    plt.xlabel('Time Step', fontsize=18)
    plt.ylabel('Belief', fontsize=18)
    # plt.annotate(s = "Agent1 reach the \n almost sure winning region", xy = (4, belief_2[4]), xytext=(2, 0.45),arrowprops=dict(facecolor='black', shrink = 0.001))
    box = ax1.get_position()
    my_xticks = np.arange(0, len(traj), 2)
    my_yticks = np.arange(0, 1.2, 0.2)
    ax1.set_xticks(my_xticks)
    ax1.set_yticks(my_yticks)
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(0.1, 1.12), ncol=3, fontsize=12)
    # plt.legend(loc='NorthOutside', fontsize = 18)
    plt.show()
