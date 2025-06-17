import numpy as np
import matplotlib.pyplot as plt
from helpers import *

def simplex_to_cartesian(w):
    #a, b, c = w
    cart_x = w[1] * 0.5 + w[2]
    cart_y = w[1] * np.sqrt(3) / 2

    return cart_x, cart_y

def apply_anchor(u, w, alpha):
    return (1 - alpha) * u + alpha * w

def plot_simplex_with_anchor(w_list, w_labels=None, alt_labels=None):
    corners = np.array([
        [0.0, 0.0],           #a
        [0.5, np.sqrt(3)/2],  #b
        [1.0, 0.0]            #c
    ])
    middles = np.array([
        [0.75, np.sqrt(3) / 4],  #op a
        [0.5, 0.0],              #op b
        [0.25, np.sqrt(3) / 4]   #op c
    ])
    thirds = np.array([
        [2/3, 1/3, 0],
        [1/3, 2/3, 0],
        [0, 2/3, 1/3],
        [0, 1/3, 2/3],
        [1/3, 0, 2/3],
        [2/3, 0, 1/3]
    ])
    
    #drawing triangle
    plt.figure(figsize=(6,6))
    for i in range(3):
        x1 = [corners[i][0], corners[(i + 1) % 3][0]]
        y1 = [corners[i][1], corners[(i + 1) % 3][1]]
        plt.plot(x1, y1, 'k-')
        x2 = [corners[i][0], middles[i][0]]
        y2 = [corners[i][1], middles[i][1]]
        plt.plot(x2, y2, 'k--', linewidth=0.6)

    str1 = alt_labels[0] if alt_labels else 'a'
    str2 = alt_labels[1] if alt_labels else 'b'
    str3 = alt_labels[2] if alt_labels else 'c'
    plt.text(-0.05, -0.05, str1, ha='right')
    plt.text(0.5, np.sqrt(3)/2 + 0.05, str2, ha='center')
    plt.text(1.05, -0.05, str3, ha='left')

    for i in range(len(thirds)):
        label = f'P{i+1}'
        x,y = simplex_to_cartesian(thirds[i])
        plt.plot(x, y, 'ko')
        if i == 1 or i == 2:
            plt.figtext(x, y, label)
            continue
        if i == 0:
            plt.text(x - 0.05, y, label, ha = 'center', va = 'center')
            continue
        if i == 3:
            plt.text(x + 0.05, y, label, ha = 'center', va = 'center')
            continue
        plt.text(x, y - 0.05, label, ha = 'center', va = 'top')
        

    for i in range(len(w_list)):
        x,y = simplex_to_cartesian(w_list[i])
        label = w_labels[i] if w_labels else f'w{i+1}'
        plt.plot(x, y, 'o', label=label)

    plt.title("Anchor Points in the Simplex")
    plt.axis('equal')
    plt.axis('off') 
    plt.legend()
    #plt.show()

def plot_change_over_alpha(w_list, u, alphas, w_labels=None, alt_labels=None, alt1=0, alt2=1):
    plt.figure()
    for i in range(len(w_list)):
        p_vals = []
        for alpha in alphas:
            shifted = apply_anchor(u, w_list[i], alpha)
            r = [0, 2, 1]
            p_vals.append(get_prob(shifted, r))
            label = w_labels[i] if w_labels else f'w{i+1}'
        plt.plot(alphas, p_vals, label=label)

    plt.xlabel("α (Influence strength)")
    plt.ylabel("P(a > c)")
    str1 = alt_labels[alt1] if alt_labels else f'candidate {alt1}'
    str2 = alt_labels[alt2] if alt_labels else f'candidate {alt2}'
    plt.title("Effect of Anchor Strength on Support for {} over {}".format(str1, str2))
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0, 1)   
    #plt.ylim(0, 1.05)
    plt.legend()
    #plt.show()




#testing functions

n = 1000
u = np.array([
        [2/3, 1/3, 0],
        [1/3, 2/3, 0],
        [0, 2/3, 1/3],
        [0, 1/3, 2/3],
        [1/3, 0, 2/3],
        [2/3, 0, 1/3]
    ])
alphas = np.linspace(0.0, 1.0, 100)
p_vals = []
w = np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.5]])

plot_simplex_with_anchor(w)
plot_change_over_alpha(w, u, alphas, alt_labels=['a','b','c'])
plt.show()
