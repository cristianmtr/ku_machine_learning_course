import numpy as np

def ex1():
    S = [0, 1]
    reps = 10000
    total = 0
    first = [np.random.choice(S)]
    rest = np.repeat(first, reps-1)
    total = np.concatenate([first, rest])
    empirical_mean = np.mean(total)
    print('empirical mean: ',empirical_mean)    

if __name__ == "__main__":
    ex1()