import numpy as np

from bayesian_inference.BayesianLearner import BayesianLearner


class Inference:
    inference_learning = BayesianLearner()

    def __init__(self):
        """
        When initialize the TRAJ, the traj should be empty
        """
        self.joint_traj = []
        self.rewards = []

    def reset_inference(self, joint_traj):
        """
        Reset the traj every time
        :return:
        """
        self.joint_traj = joint_traj
        self.update_traj(joint_traj)

    def ad_policy_prob(self, state, action):
        probs = self.inference_learning.semi_optimal_strategy(state)
        for key in probs.keys():
            if action == self.inference_learning.A[key]:
                ad_index = key
        return probs[ad_index]

    def get_ad_action(self, state):
        """
        This function will return an action given a stationary policy given the current state.
        :param state: The current state
        :return: The action
        """
        "Chose the goal according to the belief distribution"

        probs = self.inference_learning.semi_optimal_strategy(state)

        ad_a = self.inference_learning.A[np.random.choice(list(probs.keys()), p=list(probs.values()))]
        return ad_a

    def get_ad_max_action(self, state):
        probs = self.inference_learning.semi_optimal_strategy(state)

        ad_a_index = np.argmax(list(probs.values()))
        ad_name = list(self.inference_learning.A.keys())[ad_a_index]
        ad_a = self.inference_learning.A[ad_name]

        return ad_a

    def get_ad_pursuit_action(self, state):
        ctrl_state = state[0]
        ad_state = state[-1]
        "Get the current distance"
        dist = np.linalg.norm(np.array(ctrl_state) - np.array(ad_state), ord=1)

        ad_a = list(self.inference_learning.A.values())[0]

        for a in list(self.inference_learning.A.values()):
            ad_n_s = tuple(map(lambda x, y: x + y, ad_state, a))
            new_dist = np.linalg.norm(np.array(ctrl_state) - np.array(ad_n_s), ord=1)
            if new_dist < dist:
                ad_a = a
        return ad_a

    def record_reward(self, reward):
        self.rewards.append(reward)

    def get_states_seq(self, h):
        seq = []
        for state in h:
            if len(state) > 1:
                seq.append(state)
        return seq

    def update_traj(self, joint_traj, disp_flag=False):
        """
        Add the current state into the traj and update the inference
        :param joint_traj: The current joint state
        :return: NULL
        """
        self.joint_traj = joint_traj
        self.update_inference(disp_flag)

    def update_inference(self, disp_flag=False):
        """
        Update the inference according to the current traj
        :return: NULL
        """
        self.inference_learning.reset()
        self.inference_learning.Bayesian_inference(self.joint_traj, display_flag=disp_flag)

        if disp_flag is True:
            print(" ")
            print("| The current joint traj:" + str(self.joint_traj))
            "Every time we need to update the inference"

