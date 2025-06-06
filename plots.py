import numpy as np
import matplotlib.pyplot as plt
from helpers import *

def simplex_to_cartesian(w):
    #converting (w_a, w_b, w_c) to 2D Cartesian for plotting in an equilateral triangle
    a, b, c = w
    x = 0.5 * (2*b + c)  # or 0.5 * (b + 2*c)
    y = (np.sqrt(3)/2) * c
    return x, y

def plot_simplex_with_anchor(w_list, labels=None):
    corners = np.array([
        [0.5, np.sqrt(3)/2],  # a
        [0.0, 0.0],           # b
        [1.0, 0.0]            # c
    ])

    plt.figure(figsize=(6,6))
    for i in range(3):
        x = [corners[i][0], corners[(i+1)%3][0]]
        y = [corners[i][1], corners[(i+1)%3][1]]
        plt.plot(x, y, 'k-')

    plt.text(0.5, np.sqrt(3)/2 + 0.05, 'a', ha='center')
    plt.text(-0.05, -0.05, 'b', ha='right')
    plt.text(1.05, -0.05, 'c', ha='left')

    for i, w in enumerate(w_list):
        x, y = simplex_to_cartesian(w)
        label = labels[i] if labels else f'w{i+1}'
        plt.plot(x, y, 'o', label=label)

    plt.title("Anchor Points in the Simplex")
    plt.axis('equal')
    plt.axis('off')
    plt.legend()
    plt.show()

#testing functions

n = 1000
u = test_generate_utilities(n)
w = np.array([0.7, 0.2, 0.1])
alphas = np.linspace(0, 1, 100)
p_vals = []

for alpha in alphas:
    shifted = apply_anchor(u, w, alpha)
    p_vals.append(a_over_c_fraction(shifted))

plt.plot(alphas, p_vals)
plt.xlabel("α (Influence strength)")
plt.ylabel("P(a > c)")
plt.title("Probability voter prefers a over c vs. α")
plt.grid(True)
plot_simplex_with_anchor([w])
plt.show()
