import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_file(file_name):
    with open(file_name, 'rb') as fp:
        policy = pickle.load(fp)
    return policy


if __name__ == '__main__':
    winning_rates = load_file("winning_rates.pkl")
    print(' ')
    print('The times of running mcts_pomdp: ', len(winning_rates))
    print(' ')
    print('The best winning rate is: ', np.max(np.array(winning_rates)))
    print(' ')
    print('The mean is: ', np.mean(np.array(winning_rates)))
    print(' ')
    print('The variance is: ', np.var(np.array(winning_rates)))
    print(' ')
    print('The standard deviation is: ', np.std(winning_rates))

    fig = plt.figure(frameon=True)
    table = [winning_rates]
    df = pd.DataFrame(table)
    df = df.transpose()
    df.columns = ['Total Winning Rates']
    sns.set(style="whitegrid")
    sns.set(font_scale=1.5)
    ax = sns.violinplot(data=df)
    ax.set_ylabel("Winning Rate")

    plt.title("Statistic results over " + str(len(winning_rates)) + " runs")

    plt.show()
    plt.close()
