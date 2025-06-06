import numpy as np
from collections import Counter

def generate_utilities(n):
    #sampling uniformly from the 2D simplex (a, b, c utilities sum to 1)
    a = np.random.rand(n)
    b = np.random.rand(n)
    c = 1 - a - b
    mask = (c > 0)
    return np.stack((a[mask], b[mask], c[mask]), axis=1)

def test_generate_utilities(n):
    #rejection sampling from the 2D simplex
    samples = []
    while len(samples) < n:
        a, b = np.random.rand(2)
        c = 1 - a - b
        if c >= 0:
            samples.append([a, b, c])
    return np.array(samples)

def get_rankings(u):
    return np.argsort(-u, axis=1)

def apply_anchor(u, w, alpha):
    return (1 - alpha) * u + alpha * w

def count_profiles(rankings):
    labels = ['a', 'b', 'c']
    return Counter(''.join(labels[i] for i in rank) for rank in rankings)

def a_over_c_fraction(u):
    return np.mean(u[:, 0] > u[:, 2])
