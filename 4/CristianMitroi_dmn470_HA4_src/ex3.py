from matplotlib import pyplot as plt
import itertools


def ex3_1():
    print("ex 3.1...")
    # all possible labelings
    plt.clf()
    labelings = list(itertools.product([-1, 1], repeat=3))
    x = [1, 2, 2]
    y = [3, 1, 3]
    fig, ax = plt.subplots(4, 2)
    _ = fig.suptitle("All possible labelings for n=3, (blue=+1,red=-1)")
    for i, labeling in enumerate(labelings):
        c = ["b" if y == 1 else "r" for y in labeling]
        curr_ax = ax[i//2, i % 2]
        curr_ax.scatter(x, y, c=c, s=0.7)
        curr_ax.set_ylim([-1, 6])
        curr_ax.set_xlim([-1, 5])
        # curr_ax.set_markersize(0.1)
    plt.gcf().savefig('ex_3_1.png', dpi=300, bbox_inches="tight")


def ex3_2():
    print("ex 3.2...")
    plt.clf()
    # all possible labelings
    labelings = list(itertools.product([-1, 1], repeat=4))
    x = [1, 2, 2, 3]
    y = [3, 2, 1, 3]
    fig, ax = plt.subplots(4, 4)
    _ = fig.suptitle("All possible labelings for n=4, (blue=+1,red=-1)")
    for i, labeling in enumerate(labelings):
        c = ["b" if y == 1 else "r" for y in labeling]
        curr_ax = ax[i//4, i % 4]
        curr_ax.scatter(x, y, c=c, s=0.7)
        curr_ax.set_ylim([0, 4])
        curr_ax.set_xlim([0, 4])
        # curr_ax.axis('equal')
    plt.gcf().savefig('ex_3_2.png', dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    ex3_1()
    ex3_2()
