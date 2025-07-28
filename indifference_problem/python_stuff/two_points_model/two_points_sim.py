import matplotlib
import numpy as np
from scipy.integrate import quad
from tqdm import tqdm
from simulate_election_with_turnout import *


def turnout_func(u):
    return np.abs(2 * u - 1)
    #return (2 * u - 1) ** 2

def find_examples(w, tests=1000):
    successes = []
    for i in tqdm(range(tests)):
        u_a = 1/2 * np.random.rand()
        gap = 1/2 - u_a
        u_b = 1 - gap * np.random.rand()
        p_b = 1/2 * np.random.rand()
        p_a = 1 - p_b

        ex_a_votes = w(u_a) * p_a
        ex_b_votes = w(u_b) * p_b
        if p_a > p_b and ex_a_votes < ex_b_votes:
            successes.append((u_a, u_b, p_a))
    return successes

if __name__ == "__main__":
    examples = find_examples(turnout_func)
    for e in examples:
        print("u_a: {}, u_b: {}, p_a: {}".format(e[0], e[1], e[2]))

    



