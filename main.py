# coding=utf-8
import os
import pickle

import numpy as np

from ad_env import AdversarialEnvironment
from ctrl_env import ControllableEnvironment
from mcts.arg import ARG
from mcts.history import History
from mcts.mcts import MonteCarloTreeSearch
from mcts.node import Node
from simulation import Evaluation


def save_object(obj, file_name):
    """
    Save the object to a file
    :param obj: the object needs to be saved
    :param file_name: the path of the file
    :return: NULL
    """
    with open(file_name, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


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

    levels = ARG.levels  # this needs to be smaller than the number defined in the state
    num_sims = ARG.num_sims

    'Global Variables'
    # sure_winning_regions = load_sure_winning_regions("pre_compute/sure_winning_regions.pkl")
    ###20180618
    sure_winning_regions = load_sure_winning_regions("pre_compute/sure_winning_regions.pkl")
    # surewinning2 = load_sure_winning_regions("pre_compute/Set_(6,7).pkl")

    # sure_winning_regions = surewinning1.union(surewinning2)
    ###20180618
    "Create two environments"
    ctrl_env = ControllableEnvironment()
    ad_env = AdversarialEnvironment()  # the goal of the ad env is not important here

    all_policies = {}
    win_rate = []

    eva = Evaluation(ctrl_env, ad_env)
    root_state = (ctrl_env.I, ad_env.I)
    root_h = [root_state]

    "create a tree"
    tree = MonteCarloTreeSearch(sure_winning_regions, ctrl_env, ad_env)

    for i in range(ARG.num_policy):
        "reset the Monte Carlo Tree Search"
        tree.reset_tree()
        print(" ")
        print("----------------------------------------------------------------")
        print(" ")
        print("| The root state value is", root_state)
        current_node = Node(History([root_h], levels, sure_winning_regions, ctrl_env, ad_env))

        tree.uct_search(num_sims / float(levels), current_node)  # create the current root
        pi = tree.pi
        print(" ")
        print("| The size of the current MCTS is: ", len(pi))
        all_policies[i] = pi
        win_rate.append(eva.evaluate(pi, sure_winning_regions))

    "save the policy"
    f_policy = "all_policies.pkl"
    print('')
    print('policy saved to file: ', f_policy)
    save_object(all_policies, f_policy)

    "save the winning rates"
    rate_files = "winning_rates.pkl"
    print('')
    print('winning rates saved to file: ', rate_files)
    save_object(win_rate, rate_files)

    print(' ')
    print("The total winning rates: ", win_rate)
    print('The mean is :', np.mean(win_rate))
    print('The variance is: ', np.var(win_rate))
    print('The best winning rate is:', np.max(win_rate))
