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


def load_pre_computed_policy(contr_pi):
    print(' ')
    print('The size of the policy:', len(contr_pi))
    return contr_pi

def load_sure_winning_regions(sure_winning_regions):
    reg = []
    for s in sure_winning_regions:
        s1 = s[0]
        s2 = s[1]
        j_s = s1 + s2
        reg.append(j_s)

    print(" ")
    print("| The size of the sure winning regions is: " + str(len(reg)))
    return reg

def simulate(sureWinReg, sureWinPolicy, targetSet, initState, traj, inference):

    max_time_steps = ARG.levels
    simulation_times = ARG.simulation_times

    levels = ARG.levels  # this needs to be smaller than the number defined in the state
    num_sims = ARG.num_sims

    'Global Variables'
    sure_winning_regions = load_sure_winning_regions(sureWinReg)
    sure_winning_policy = load_pre_computed_policy(sureWinPolicy)

    "Create two environments"
    ctrl_env = ControllableEnvironment()
    ad_env = AdversarialEnvironment()

    root_state = initState
    s = root_state
    "create a tree"
    tree = MonteCarloTreeSearch(sure_winning_regions, ctrl_env, ad_env)
    inference_online =inference

    print(" ")
    print("| This state is: ", s)
    contr_s = s[0]
    ad_s = s[1]
    joint_traj = traj
    actions = []
    inference_online.reset_inference(joint_traj)
    print(' ')
    print("belief:", inference_online.inference_learning.P_h_g)

    touch = False
    if np.linalg.norm(np.array(contr_s) - np.array(ad_s), ord=1) <= 1:
        print(' ')
        print('| Sorry! The Blue agent is caught!')

    elif contr_s in targetSet:
        print(' ')
        print('| Congrats!!! The blue agent reaches the temp goal!!!')
        print('------------------------------------')

    elif contr_s + ad_s in sure_winning_regions:
        if touch is False:
            print(' ')
            print('| Congrats!!! The blue agent reaches the sure winning regions!!!')
            print('------------------------------------')
            touch = True

        contr_a = ctrl_env.A_full[sure_winning_policy[(contr_s, ad_s)]]

    else:
        current_node = Node(History([joint_traj], levels, sure_winning_regions, ctrl_env, ad_env))
        tree.reset_tree()
        best_child = tree.uct_search(num_sims / float(levels), current_node)  # create the current root

        contr_a = best_child.history.value[-1]

    contr_n_s = ctrl_env.sto_trans(contr_s, tuple(contr_a))

    ad_a = inference_online.get_ad_action(s)
    # ad_a = inference_online.get_ad_max_action(s)
    # ad_a = inference_online.get_ad_pursuit_action(s)

    ad_n_s = ad_env.sto_trans(ad_s, ad_a)
    contr_s = contr_n_s
    ad_s = ad_n_s
    s = (contr_n_s, ad_n_s)

    print(" ")
    print("| The current state is :", s)
    joint_traj.append(s)
    inference_online.update_traj(joint_traj, disp_flag=False)
    print(' ')
    print(inference_online.inference_learning.P_h_g)
    # print(inference_online.inference_learning.total_KL)
    actions.append(contr_a)
    return s, joint_traj, inference_online
