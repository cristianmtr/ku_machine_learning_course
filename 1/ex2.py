import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


def ex2():
    nr_reps = 1000000
    options = [0, 1]  # tails, head
    simulations = np.random.choice(options, (nr_reps, 20))
    sum_freq = np.sum(simulations, axis=1)/20

    alphas = [i for i in range(50, 105, 5)]

    alphas = np.array(alphas)/100

    final_freqs = np.repeat(0, alphas.shape)

    for freq in sum_freq:
        for i in range(len(alphas)):
            if i != len(alphas)-1:
                lower_alpha = alphas[i]
                higher_alpha = alphas[i+1]
                if freq >= lower_alpha and freq < higher_alpha:
                    final_freqs[i] += 1
            elif i == len(alphas)-1:
                if freq == alphas[i]:
                    final_freqs[i] += 1

    final_freqs_perc = final_freqs/nr_reps

    plt.scatter(alphas, final_freqs_perc)
    plt.xlabel('alpha')
    plt.ylabel('Frequency')
    plt.xticks([i for i in np.arange(0.5, 1.5, 0.05)])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    # 2.4
    plt.plot(np.arange(0.5, 1.02, 0.05), 0.5 /
             np.arange(0.5, 1.02, 0.05), 'r', label='Markov')

    # 2.5
    plt.plot(np.arange(0.5, 1.02, 0.05), (np.var(options)/20) /
             ((np.arange(0.5, 1.02, 0.05)-0.5)**2), 'g', label='Chebyshev')
    plt.legend(loc='upper right')
    plt.ylim((-0.1, 1.1))
    plt.gcf().savefig('ex2.png')


if __name__ == "__main__":
    ex2()
