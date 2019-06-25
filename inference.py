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

    def reset_inference(self, joint_traj, d):
        """
        Reset the traj every time
        :return:
        """
        self.inference_learning.reset()
        self.update_traj(joint_traj, d)

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

    # def get_ad_max_action(self, state):
    #     """
    #     This function will return an action given a stationary policy given the current state.
    #     :param state: The current state
    #     :return: The action
    #     """
    #     "Chose the goal according to the belief distribution"
    #
    #     probs = self.inference_learning.semi_optimal_strategy(state)
    #
    #     ad_a_index = np.argmax(list(probs.values()))
    #     ad_name = list(self.inference_learning.A.keys())[ad_a_index]
    #     ad_a = self.inference_learning.A[ad_name]
    #
    #     return ad_a

    def record_reward(self, reward):
        self.rewards.append(reward)

    def update_traj(self, joint_traj, d, disp_flag=False):
        """
        Add the current state into the traj and update the inference
        :param joint_traj: The current joint state
        :return: NULL
        """
        if d is False:
            self.joint_traj = joint_traj
            self.update_inference(disp_flag)

    def update_inference(self, disp_flag=False):
        """
        Update the inference according to the current traj
        :return: NULL
        """

        if disp_flag is True:
            print(" ")
            print("| The current joint traj:" + str(self.joint_traj))
            "Every time we need to update the inference"

            self.inference_learning.Bayesian_inference(self.joint_traj, display_flag=disp_flag)
