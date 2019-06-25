# coding=utf-8
import os
import pickle

import numpy as np

from ad_env import AdversarialEnvironment
from ctrl_env import ControllableEnvironment
from inference import Inference
from mcts.arg import ARG
from mcts.history import History
from mcts.mcts import MonteCarloTreeSearch
from mcts.node import Node
from simulation import Evaluation


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


if __name__ == '__main__':

    max_time_steps = ARG.levels
    simulation_times = ARG.simulation_times

    levels = ARG.levels  # this needs to be smaller than the number defined in the state
    num_sims = ARG.num_sims

    'Global Variables'
    sure_winning_regions = load_sure_winning_regions("pre_compute/sure_winning_regions.pkl")

    "Create two environments"
    ctrl_env = ControllableEnvironment()
    ad_env = AdversarialEnvironment()

    all_policies = {}
    win_rate = []

    eva = Evaluation(ctrl_env, ad_env)
    root_state = (ctrl_env.I, ad_env.I)

    "create a tree"
    tree = MonteCarloTreeSearch(sure_winning_regions, ctrl_env, ad_env)

    sure_winning_policy = load_pre_computed_policy('pre_compute/almostSureWinningPolicy.pkl')

    inference_online = Inference()

    win = 0
    caught = 0
    touch_win = 0

    i = 0

    while i < simulation_times:
        i += 1

        counter = 0

        # s = (ctrl_env.I, ad_env.I)
        s = ((0, 5), (10, 5))  # good initial location

        print(" ")
        print("| The initial state is: ", s)
        contr_s = s[0]
        ad_s = s[1]

        joint_traj = [s]
        actions = []

        inference_online.reset_inference(joint_traj, False)

        touch = False

        while counter < max_time_steps:

            if contr_s == (6, 3) or contr_s == (6, 7):
                win += 1
                print(' ')
                print('| Congrats!!! The blue agent reaches the true goal!!!')
                print('------------------------------------')
                break

            elif np.linalg.norm(np.array(contr_s) - np.array(ad_s), ord=1) <= 1:
                caught += 1
                print(' ')
                print('| Sorry! The Blue agent is caught!')
                break

            elif contr_s + ad_s in sure_winning_regions:
                inference_online.update_traj(joint_traj, False, disp_flag=False)
                if touch is False:
                    touch_win += 1
                    print(' ')
                    print('| Congrats!!! The blue agent reaches the sure winning regions!!!')
                    print('------------------------------------')
                    touch = True

                contr_a = inference_online.inference_learning.A_full[sure_winning_policy[(contr_s, ad_s)]]

            else:
                inference_online.update_traj(joint_traj, False, disp_flag=False)

                current_node = Node(History([joint_traj], levels, sure_winning_regions, ctrl_env, ad_env))

                best_child = tree.uct_search(num_sims / float(levels), current_node)  # create the current root

                contr_a = best_child.history.value[-1]

            contr_n_s = ctrl_env.sto_trans(contr_s, tuple(contr_a))

            ad_a = inference_online.get_ad_action(s)
            ad_n_s = ad_env.sto_trans(ad_s, ad_a)

            counter += 1
            contr_s = contr_n_s
            ad_s = ad_n_s
            s = (contr_n_s, ad_n_s)

            print(" ")
            print("| The current state is :", s)
            joint_traj.append(s)
            actions.append(contr_a)

    print(' ')
    print('| The winning percentage is: ', win / simulation_times)

    print(' ')
    print('| The touch winning regions percentage is: ', touch_win / simulation_times)

    print(' ')
    print('| The caught percentage is: ', caught / simulation_times)
