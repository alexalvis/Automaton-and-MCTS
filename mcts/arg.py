import numpy as np


class ARG:
    levels = 20
    iterations = 2000
    num_sims = levels * iterations
    true_belief = np.array([0, 1, 0])
    decoy_set = [(6, 0), (9, 5), (6, 10)]

    "mcts_pomdp scalar.  Larger scalar will increase exploitation, smaller will increase exploration."
    SCALAR = 1 / np.sqrt(2.0)  # satisfy the Hoeffding inequality with rewards in the range [0,1]

    num_policy = 1
    simulation_times = 100.0
