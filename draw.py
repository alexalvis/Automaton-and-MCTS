import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

if __name__ == '__main__':
    filename = "reward_mat.pkl"
    with open(filename, "rb") as f1:
        reward_mat = pickle.load(f1)
    Y = np.empty((100,10))
    matrix = []
    for i in range(100):
        matrix.append(reward_mat[i])
    matrix = np.array(matrix)
    matrix_mean = np.mean(matrix, axis = 0)
    matrix_max = np.max(matrix, axis = 0)
    matrix_min = np.min(matrix, axis = 0)
    matrix_std = np.std(matrix, axis = 0)
    fig, ax = plt.subplots()
    ax.plot(matrix_mean, linestyle = '-', label = "mean", color = "red")
    ax.fill_between(range(len(matrix[0])), matrix_mean+matrix_std, matrix_mean - matrix_std, color = "red", alpha = 0.25)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=3, fontsize=12)
    plt.grid(True)
    plt.xlabel('Iterations', fontsize = 18)
    plt.ylabel('Total reward', fontsize = 18)
    plt.show()
