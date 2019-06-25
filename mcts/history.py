import random
from copy import copy

import numpy as np


class History:

    def __init__(self, value, turn, sure_winning_regions, ctrl_env, ad_env):
        """
        Initialize a history
        :param value
        :param turn:
        :param sure_winning_regions:
        """

        self.value = value

        self.turn = turn  # defined as the depth
        self.sure_winning_regions = sure_winning_regions

        self.ctrl_env = ctrl_env
        self.ad_env = ad_env

        self.num_moves, self.d = self.get_num_moves(value)

        return

    def sample(self, inference):
        if self.d is True:
            new_history = self.next_belief(inference)  # just expand one
        else:
            new_history = self.next_action()

        return new_history

    def get_num_moves(self, value):
        h = self.value[0]
        if len(value) > 1:
            if self.ad_env.rand > 0:
                current_state = h[-2]
                ad_state = current_state[-1]
                neighbors = self.ad_env.find_neighbors(ad_state)
                d = True
            else:
                return 1, True
        else:
            current_state = h[-1]
            ctrl_state = current_state[0]
            neighbors = self.ctrl_env.find_neighbors(ctrl_state)
            d = False
        return len(neighbors), d

    def next_action(self):
        h = self.value[0]
        current_state = h[-1]
        ctrl_state = current_state[0]

        ctrl_action = random.choice([x for x in self.ctrl_env.A])

        next_ctrl_state = self.ctrl_env.sto_trans(ctrl_state, ctrl_action)

        new_h = copy(h)

        new_h.append((next_ctrl_state,))

        next_history = History([new_h, ctrl_action], self.turn - 1, self.sure_winning_regions, self.ctrl_env, self.ad_env)
        return next_history

    def next_belief(self, inference):
        """
        The next state is selected by the random policy
        :return:
        """
        "Get the current state of controllable agent and ad agent"
        h = self.value[0]
        ctrl_action = self.value[1]

        current_state = h[-2]
        ctrl_state = h[-1][0]
        ad_state = h[-2][-1]

        "get the ad action according to the policy"
        ad_a = inference.get_ad_action(current_state)

        ad_next_state = self.ad_env.sto_trans(ad_state, ad_a)

        "Now the adversary is moving also"
        next_state = (ctrl_state, ad_next_state)

        new_h = copy(h)
        new_h.append(next_state)

        next_belief = History([new_h, ], self.turn - 1, self.sure_winning_regions, self.ctrl_env, self.ad_env)
        return next_belief

    def terminal(self, terminalState, disp_flag=False):
        """
        Determine the current node.py is the terminal node.py based on the turn of the state
        :return: True, if the current state is in the terminal node.py. False, otherwise.
        """
        h = self.value[0]
        current_state = h[-1]
        ctrl_state = current_state[0]

        "if the state reaches the true goal or turn is over or adversary reaches the true goal or "
        "state reaches sure winning region or sure losing regions"

        if self.d is False:
            ad_state = current_state[1]

            if np.linalg.norm(np.array(ctrl_state) - np.array(ad_state), ord=1) <= 1:
                if disp_flag is True:
                    print(" ")
                    print("| The system is caught!!!")
                return True
            elif ctrl_state + ad_state in self.sure_winning_regions:
                if disp_flag is True:
                    print(" ")
                    print("| The system touches the sure winning region!!!")
                return True
            elif self.turn == 0:
                if disp_flag is True:
                    print(" ")
                    print("| The system reaches the maximum levels!!!")
                return True

            elif ctrl_state in terminalState:
                if disp_flag is True:
                    print(" ")
                    print("| The system touches true goal!!!")
                return True

        return False

    def reward(self, inference):
        """
        Return the immediate reward with the KL divergence.
        :return: the immediate reward with KL divergence, if the system reaches the goal. 0, otherwise.
        """
        h = self.value[0]
        current_state = h[-1]
        ctrl_state = current_state[0]
        ad_state = current_state[1]

        r = 0

        kl_reward = inference.inference_learning.total_KL  # add the kl divergence

        r += kl_reward

        # "if the joint state is the in the sure_winning_regions, it will have the reward."
        if ctrl_state + ad_state in self.sure_winning_regions:
            r += 100

        # if ctrl_state == (6, 3) or ctrl_state == (6, 7) and np.linalg.norm(np.array(ctrl_state) - np.array(ad_state), ord=1) > 1:
        #     r += 100

        return r

    def __eq__(self, other):
        if self.value == other.value:
            return True
        return False

    def __repr__(self):
        s = "The value of the current belief is: %s; " % (str(self.value))
        return s
