import numpy as np

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

def solve_linear_threshold(u, w):
    u = np.array(u, dtype=float)
    w = np.array(w, dtype=float)
    # a > c
    alpha1 = threshold_for_inequality(u[0], u[2], w[0], w[2])
    # c > b
    alpha2 = threshold_for_inequality(u[2], u[1], w[2], w[1])
    return max(alpha1, alpha2)

def get_best_point(u, resolution=100):
    lambdas = np.linspace(0.0, 1.0, resolution, endpoint=True)
    min_alpha = 1
    opt_w = None
    for l in lambdas:
        w = [(1 + l) / 2, 0, (1 - l) / 2]
        alpha = solve_linear_threshold(u, w)
        if alpha < min_alpha:
            min_alpha = alpha
            opt_w = w
    return min_alpha, opt_w

P1 = [2/3, 1/3, 0]
P2 = [1/3, 2/3, 0]
P3 = [0, 2/3, 1/3]
P4 = [0, 1/3, 2/3]
P5 = [1/3, 0, 2/3]

print(get_best_point(P3))

