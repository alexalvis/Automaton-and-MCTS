import os
import pickle

import numpy as np

from inference import Inference
from mcts.arg import ARG


def load_pre_computed_policy(filename):
    with open(filename, 'rb') as input:
        contr_pi = pickle.load(input)

    print(' ')
    print('The size of the policy:', len(contr_pi))
    return contr_pi


def load_sure_winning_regions(file_name):
    DictList = []
    with open(file_name, "rb") as f:
        fileSize = os.fstat(f.fileno()).st_size
        while f.tell() < fileSize:
            DictList.append(pickle.load(f))
    reg = []
    sure_winning_regions = DictList[-1]
    for s in sure_winning_regions:
        s1 = s[0]
        s2 = s[1]
        j_s = s1 + s2
        reg.append(j_s)

    print(" ")
    print("| The size of the sure winning regions is: " + str(len(reg)))
    return reg


class Evaluation:

    def __init__(self, ctrl_env, ad_env):
        self.ctrl_env = ctrl_env
        self.ad_env = ad_env

        self.max_time_steps = ARG.levels
        self.simulation_times = ARG.simulation_times

    def get_ad_action(self, state, inference):
        """
        This function will return an action given a stationary policy given the current state.
        :param state: The current state
        :return: The action
        """
        "Chose the goal according to the belief distribution"

        probs = inference.inference_learning.semi_optimal_strategy(state)

        ad_a = inference.inference_learning.A[np.random.choice(list(probs.keys()), p=list(probs.values()))]
        return ad_a

    def evaluate(self, policy, regions):

        ctrl_pi = policy
        sure_winning_regions = regions
        sure_winning_policy = load_pre_computed_policy('pre_compute/almostSureWinningPolicy.pkl')

        inference = Inference()

        win = 0
        caught = 0

        i = 0

        while i < self.simulation_times:
            i += 1

            counter = 0

            s = (self.ctrl_env.I, self.ad_env.I)
            contr_s = s[0]
            ad_s = s[1]

            joint_traj = [s]

            inference.reset_inference(joint_traj, False)

            touch = False

            while counter < self.max_time_steps:
                print(" ")
                print("| This is the ", counter, "-th step")

                if np.linalg.norm(np.array(contr_s) - np.array(ad_s), ord=1) <= 1:
                    caught += 1
                    print(' ')
                    print('| Sorry! The Blue agent is caught!')
                    break

                elif contr_s == (6, 3) or contr_s == (6, 7):
                    win += 1
                    print(' ')
                    print('| Congrats!!! The blue agent reaches the true goal!!!')
                    print('------------------------------------')
                    break

                elif contr_s + ad_s in sure_winning_regions:
                    inference.update_traj(joint_traj, False, disp_flag=False)
                    if touch is False:
                        print(' ')
                        print('| Congrats!!! The blue agent reaches the sure winning regions!!!')
                        print('------------------------------------')
                        touch = True
                    print(contr_s, ad_s)
                    contr_a = inference.inference_learning.A_full[sure_winning_policy[(contr_s, ad_s)]]

                else:
                    inference.update_traj(joint_traj, False, disp_flag=False)

                    probs = []
                    for a in self.ctrl_env.A:
                        try:
                            probs.append(ctrl_pi[a, tuple(joint_traj)])  # get the probability distribution on according to NCTS
                        except KeyError:
                            print(' ')
                            print('| Have not see this history before. ')
                            probs.append(0)
                    if sum(probs) == 0:
                        probs = [0.25] * 4

                    probs = probs / np.linalg.norm(probs, ord=1)

                    contr_a = self.ctrl_env.A[np.random.choice(range(len(self.ctrl_env.A)), p=probs)]

                contr_n_s = self.ctrl_env.sto_trans(contr_s, contr_a)

                ad_a = self.get_ad_action(s, inference)
                ad_n_s = self.ad_env.sto_trans(ad_s, ad_a)

                counter += 1
                contr_s = contr_n_s
                ad_s = ad_n_s
                s = (contr_n_s, ad_n_s)
                joint_traj.append(s)

        return win / self.simulation_times
