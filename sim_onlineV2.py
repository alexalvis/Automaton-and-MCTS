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

def simulate(sureWinReg, sureWinPolicy, targetSet, initState):

    max_time_steps = ARG.levels
    simulation_times = ARG.simulation_times

    levels = ARG.levels
    num_sims = ARG.num_sims

    sure_winning_regions = load_sure_winning_regions(sureWinReg)
    sure_winning_policy = load_pre_computed_policy(sureWinPolicy)

    ctrl_env = ControllableEnvironment()
    ad_env = AdversarialEnvironment()

    eva = Evaluation(ctrl_env, ad_env)
    root_state = initState

    "create a tree"
    tree = MonteCarloTreeSearch(sure_winning_regions, ctrl_env, ad_env)
    inference_online = Inference()

    s = root_state
    print("| The initial state is: ", s)
    contr_s = s[0]
    ad_s = s[1]
    joint_traj = [s]
    actions = []
    inference_online.reset_inference(joint_traj, False)
    touch = False
    redo = True
    touch_win = 0
    while redo == True:
        counter = 0
        while counter < max_time_steps:
            if np.linalg.norm(np.array(contr_s) - np.array(ad_s), ord=1) <= 1:
                print(' ')
                print('| Sorry! The Blue agent is caught!')
                print("redo the simulate")
                break

            elif contr_s in targetSet:
                print(' ')
                print('| Congrats!!! The blue agent reaches the true goal!!!')
                print('------------------------------------')
                redo = False
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
    return s

