import numpy as np

def threshold_for_inequality(ua, ub, wa, wb):
    P = ua - ub
    Q = wa - wb
    #if P > 0 already always true
    if P > 0:
        return 0.0
    #if P <= 0, need P + alpha * (Q - P) > 0 --> alpha > -P / (Q - P)
    denom = Q - P
    if denom == 0:
        #if Q = P and P <= 0, never true
        return 1.0 if P <= 0 else 0.0
    alpha = -P / denom
    return np.clip(alpha, 0.0, 1.0)

def solve_linear_threshold(u, w, i=0, j=2, k=1):
    #default checks a > c > b
    u = np.array(u, dtype=float)
    w = np.array(w, dtype=float)
    # a > c
    alpha1 = threshold_for_inequality(u[i], u[j], w[i], w[j])
    # c > b
    alpha2 = threshold_for_inequality(u[j], u[k], w[j], w[k])
    return max(alpha1, alpha2)

def find_min_alpha(u, resolution=100):
    #only works for a > c > b region! identifies optimal points
    lambdas = np.linspace(0.0, 1.0, resolution, endpoint=True)
    min_alpha = 1
    opt_w = None
    for l in lambdas:
        w = [(1 - l) / 2 + l, 0, (1 - l) / 2]
        alpha = solve_linear_threshold(u, w)
        if alpha < min_alpha:
            min_alpha = alpha
            opt_w = w
    return min_alpha, opt_w

def find_min_alpha_full(u, resolution=100):
    lambdas = np.linspace(0.0, 1.0, resolution, endpoint=True)
    min_alpha = 1
    opt_w = None
    for l in lambdas:
        w1 = [(1 - l) / 2 + l, 0, (1 - l) / 2]
        w2 = [(1 - l) / 2 + l / 3, l / 3, (1 - l) / 2 + l / 3]
        w3 = [l / 3 + (1 - l), l / 3, l / 3] 
        alphas = [(solve_linear_threshold(u, w1), w1), (solve_linear_threshold(u, w2), w2), (solve_linear_threshold(u, w3), w3)]
        for alpha, w in alphas:
            if alpha < min_alpha:
                min_alpha = alpha
                opt_w = w
    return min_alpha, opt_w

def find_max_dot(u, resolution=100):
    lambdas = np.linspace(0.0, 1.0, resolution, endpoint=True)
    max_dot = -np.inf
    opt_w = None
    verts = [[1 / 2, 0, 1 / 2], [1, 0, 0], [1 / 3, 1 / 3, 1 / 3]]
    for l in lambdas:
        #explicitly test vertices bc often optimal
        w1 = [(1 - l) / 2 + l, 0, (1 - l) / 2]
        w2 = [(1 - l) / 2 + l / 3, l / 3, (1 - l) / 2 + l / 3]
        w3 = [l / 3 + (1 - l), l / 3, l / 3]
        W = [verts[0], verts[1], verts[2], w1, w2, w3]
        for w in W:
            d = np.dot(u, w)
            if d > max_dot + .000001:
                max_dot = d
                opt_w = w
    return max_dot, opt_w

P1 = [2/3, 1/3, 0]
P2 = [1/3, 2/3, 0]
P3 = [0, 2/3, 1/3]
P4 = [0, 1/3, 2/3]
P5 = [1/3, 0, 2/3]

P = [P1, P2, P3, P4, P5]

#for p in P:
    #print(p, find_min_alpha_full(p), find_max_dot(p), '\n')


'''
def WRONG_threshold_for_inequality(ua, ub, wa, wb):
    A = ua - ub
    B = wa - wb
    if A > 0 and B >= A:
        return 0.0
    if A <= 0 and B <= A:
        return 1.0
    denom = B - A
    if denom == 0:
        return 0.0 if A > 0 else 1.0
    alpha = -A / denom
    return np.clip(alpha, 0.0, 1.0)
'''
