import numpy as np
from collections import Counter

def generate_utilities_uniform(n):
    #rejection sampling from the 2D simplex
    samples = []
    while len(samples) < n:
        a, c = np.random.rand(2)
        b = 1 - a - c
        if b >= 0:
            samples.append([a, b, c])
    return np.array(samples)

def get_rankings(u):
    return np.argsort(-u, axis=1)

def apply_anchor(u, w, alpha):
    return (1 - alpha) * u + alpha * w

def simplex_to_cartesian(w):
    #a, b, c = w
    cart_x = w[1] * 0.5 + w[2]
    cart_y = w[1] * np.sqrt(3) / 2
    return cart_x, cart_y  

def count_profiles(rankings):
    labels = ['a', 'b', 'c']
    return Counter(''.join(labels[i] for i in rank) for rank in rankings)

def pairwise_fraction(u, i, j):
    return np.mean(u[:, i] > u[:, j])

def p_acb(u, alt1=0, alt2=2, alt3=1):
    return np.mean((u[:, alt1] > u[:, alt2]) & (u[:, alt2] > u[:, alt3]))

def get_prob(u, ranking):
    if len(ranking) == 1:
        i = ranking[0]
        return np.mean((u[:, i] > u[:, (i + 1) % 3]) & (u[:, i] > u[:, (i + 2) % 3]))
    elif len(ranking) == 2:
        i, j = ranking
        return np.mean(u[:, i] > u[:, j])
    elif len(ranking) == 3:
        i, j, k = ranking
        return np.mean((u[:, i] > u[:, j]) & (u[:, j] > u[:, k]))
    return []




'''
def wrong_generate_utilities(n):
    #sampling uniformly from the 2D simplex (a, b, c utilities sum to 1)
    a = np.random.rand(n)
    b = np.random.rand(n)
    c = 1 - a - b
    mask = (c > 0)
    return np.stack((a[mask], b[mask], c[mask]), axis=1)
'''
