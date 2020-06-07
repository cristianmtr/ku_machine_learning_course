import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


def ex1():
    nr_reps = 1000000
    options = [0, 1]  # tails, head
    simulations = np.random.choice(options, (nr_reps, 20))
    sum_freq = np.sum(simulations, axis=1)/20

    alphas = [i for i in range(50, 105, 5)]

    alphas = [round(a,2) for a in np.array(alphas)/100]

    final_freqs = np.repeat(0, len(alphas))

    freqs = []
    for a in alphas:
        nr_of_instances = len(sum_freq[sum_freq>=a])
        freqs.append(nr_of_instances/nr_reps)

    plt.scatter(alphas, freqs, c='b', label="empirical data")

    # 2.4
    plt.plot(np.arange(0.5, 1.02, 0.05), 0.5 /
             np.arange(0.5, 1.02, 0.05), 'r', label='Markov')

    # 2.5
    chebyshev_values = (np.var(options)/20) / ((np.arange(0.5, 1.02, 0.05)-0.5)**2)
    plt.plot(np.arange(0.5, 1.02, 0.05),
             chebyshev_values, 'g', label='Chebyshev')

    hoeffding_values = np.e**(-2*20*(np.arange(0.5, 1.02, 0.05)-0.5)**2)
    print('hoeffding values for 0.95 and 1: ', hoeffding_values[-2:])
    plt.plot(np.arange(0.5, 1.02, 0.05),
             hoeffding_values, 'purple', label='Hoeffding')

    # plot it
    plt.legend(loc='upper right')
    plt.xlabel('alpha')
    plt.ylabel('Frequency')
    plt.xticks([i for i in np.arange(0.5, 1.5, 0.05)])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylim((-0.1, 1.1))
    plt.xlim((0.5, 1.02))
    plt.gcf().savefig('ex1.png')


if __name__ == "__main__":
    ex1()
