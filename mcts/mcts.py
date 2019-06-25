import matplotlib.pyplot as plt
import numpy as np

from inference import Inference
from mcts.arg import ARG


def normalize(d):
    factor = 1.0 / sum(d.values())
    for k in d:
        d[k] = d[k] * factor
    return d


class MonteCarloTreeSearch:
    def __init__(self, sure_winning_regions, ctrl_env, ad_env):

        self.pi = {}  # the converged policy

        self.sure_winning_regions = sure_winning_regions
        self.ctrl_env = ctrl_env
        self.ad_env = ad_env
        self.inference = None

    def reset_tree(self):
        self.pi = {}

    def uct_search(self, budget, root, targetState, disp_flag=False):
        """
        Monte Carlo Tree Search with UCB (Upper confidence Bound) tree selection policy
        :param budget: given the computational resource
        :param root: the current state
        :return: The best child
        """
        inference = Inference()
        win = 0
        limit = ARG.iterations - 1000     #1000 here
        for k in range(int(budget)):  # simulate within computational budget
            if disp_flag is True:
                print(" ")
                print("This is the ", k, "th iterations")

            inference.reset_inference(root.history.value[0], root.history.d)    #value[0] is the ctrl_state,

            inference, front = self.tree_policy(root, targetState, inference)

            inference, reward = self.default_policy(front.history, targetState, inference)

            if k > limit and reward > 99:
                win += 1

            self.back_up(front, reward)

            inference.record_reward(reward)

        print(' ')
        print(" After", limit, "simulation, the winning rate is: ", win / (int(k) - limit))

        # plt.rcParams.update({'font.size': 18})
        # plt.figure()
        # plt.xlabel('Iteration count')
        # plt.ylabel('Rewards')
        # plt.plot(range(len(inference.rewards)), inference.rewards)
        # plt.show()

        self.inference = inference

        return self.best_child(root, 0, inference)

    def tree_policy(self, node, targetState, inference):
        """
        Select or create a leaf node from the nodes already contained within the search tree (selection and expansion)
        :param node: the root node
        :return: the last node reached during the tree policy
        """
        "If the state is not false"
        while node.history.terminal(targetState) is False:
            if node.fully_expanded() is False:
                child_node = self.expand(node, inference)
                inference.update_traj(child_node.history.value[0], child_node.history.d)
                return inference, child_node
            else:
                new_node = self.best_child(node, ARG.SCALAR, inference)
                inference.update_traj(new_node.history.value[0], new_node.history.d)
                node = new_node
        return inference, node

    def expectation(self, hb, child_nodes, inference):
        h = hb.history.value[0]
        current_state = h[-2]
        current_ad_state = current_state[1]

        # neighbors = self.ad_env.find_neighbors(current_ad_state)
        expected_score = 0

        for c in child_nodes:
            for a in self.ad_env.A:
                policy_prob = inference.ad_policy_prob(current_state, a)
                n_s = c.history.value[0][-1][1]
                expected_score += self.ad_env.P[current_ad_state, a, n_s] * policy_prob * (c.V / c.visits)

        return expected_score

    def best_child(self, node, scalar, inference):
        """
        Choose the best child of the current based on the criterion.
        :param node: the node of the current state.
        :param scalar: the scalar in the exploration term.
        :return: the best child of the current node.
        """
        best_score = -np.Inf
        scores = []     ##619
        if node.history.d is False:
            best_children = []
            for c in node.children:
                c.V = self.expectation(c, c.children, inference)
                exploit = c.V / c.visits
                explore = np.sqrt(2.0 * np.log(node.visits) / float(c.visits))
                score = exploit + scalar * explore
                scores.append(score)     ##619
                if score == best_score:
                    best_children.append(c)
                if score > best_score:
                    best_children = [c]
                    best_score = score
            if len(best_children) == 0:
                Warning("OOPS: no best child found, probably fatal")
            # self.store_policy(scores,node)
            return np.random.choice(best_children)
        else:
            best_child = None
            h = node.history.value[0]
            current_state = h[-2]

            current_ad_state = current_state[1]
            ad_action = inference.get_ad_action(current_state)
            next_ad_state = self.ad_env.sto_trans(current_ad_state, ad_action)

            for c in node.children:
                if c.history.value[0][-1][1] == next_ad_state:
                    best_child = c

            if best_child is None:
                return c

            return best_child

    def store_policy(self, scores, node):
        exp_scores = []
        for a in scores:
            exp_scores.append(scores[a])

        probs = exp_scores / np.linalg.norm(exp_scores, ord=1)

        for iter in range(len(scores)):
            self.pi[list(scores.keys())[iter], tuple(node.history.joint_traj)] = probs[iter]

    @staticmethod
    def default_policy(history, targetState, inference):
        """
        The default policy is the random policy
        """
        while history.terminal(targetState) is False:
            history = history.sample(inference)
            inference.update_traj(history.value[0], history.d)

        return inference, history.reward(inference)

    @staticmethod
    def back_up(node, reward):
        """
        Once the terminal node is reached, the nodes along that path need to be updated.
        :param node: the terminal node
        :param reward: the immediate reward
        :return: NULL
        """
        while node is not None:
            node.update(reward)
            node = node.parent
        return

    def expand(self, node, inference):
        """
        One (or more) Child actions are added to expand the tree.
        :param node: the node of the current history
        :return: the expanded node
        """
        tried_children = [c.history for c in node.children]

        new_history = node.history.sample(inference)  # just expand one

        while new_history in tried_children:  # if the action has been tried, then keep expanding
            new_history = node.history.sample(inference)

        node.add_child(new_history)  # add the node and action to the tried children
        return node.children[-1]
