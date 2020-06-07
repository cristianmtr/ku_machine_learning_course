import math
import numpy as np


def func(p):
    return math.e**(-20000*(0.95-p)**2) * p**100


def ex4():
    p_v = np.linspace(0.0, 1, 500000)
    func_v = func(p_v)
    print('argmax', np.argmax(func_v))
    p_value = p_v[np.argmax(func_v)]
    print('p value', p_value)
    print('Event B prob', p_value**100)

if __name__ == "__main__":
    ex4()