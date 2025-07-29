import cvxpy as cp
import numpy as np

N = 500
u = np.linspace(0, 1, N)
du = u[1] - u[0]
epsilon = 1e-3


w = 0.5 * np.abs(2 * u - 1) + .5
#w = (u - 1/2) ** 2
#c = 2
#w = (c * u - c / 2) ** 2


#general f:

f = cp.Variable(N)

weighted_B = cp.sum(cp.multiply(w * (u >= 0.5), f)) * du
weighted_A = cp.sum(cp.multiply(w * (u < 0.5), f)) * du
full_A = cp.sum(cp.multiply((u < 0.5), f)) * du

constraints = [
    cp.sum(f) * du == 1,                 #normalize f to integrate to 1
    full_A >= 0.5 + epsilon,             #A wins in full turnout
    weighted_B >= weighted_A + epsilon,  #B wins in weighted turnout
    f >= 0,                              #non-negative density
]

#Lipschitz constraint (approximate)
L = 1500  #max derivative
for i in range(N - 1):
    constraints.append(cp.abs(f[i + 1] - f[i]) <= L * du)

#default problem
prob = cp.Problem(cp.Minimize(weighted_B - weighted_A), constraints)

#random directions for variation
#random_obj = cp.sum(cp.multiply(np.random.randn(N), f))
#prob = cp.Problem(cp.Maximize(random_obj), constraints)

#problem with entropy
#entropy = -cp.sum(cp.entr(f)) * du  # encourages more uniform, smoother distributions
#prob = cp.Problem(cp.Minimize(entropy), constraints)

prob.solve()


'''
#making f as flat as possible:
f = cp.Variable(N)
f_max = cp.Variable()
f_min = cp.Variable()

weighted_B = cp.sum(cp.multiply(w * (u >= 0.5), f)) * du
weighted_A = cp.sum(cp.multiply(w * (u < 0.5), f)) * du
full_A = cp.sum(cp.multiply((u < 0.5), f)) * du

constraints = [
    cp.sum(f) * du == 1,                 #normalize f to integrate to 1
    full_A >= 0.5 + epsilon,             #A wins in full turnout
    weighted_B >= weighted_A + epsilon,  #B wins in weighted turnout
    f >= 0,                              #non-negative density
    f <= f_max,
    f >= f_min,
]

alpha = 1.0    #TV penalty
beta = 100.0   #curvature penalty

total_variation = cp.sum(cp.abs(f[1:] - f[:-1]))
second_diff = f[2:] - 2 * f[1:-1] + f[:-2]
curvature_penalty = cp.sum_squares(second_diff)

objective = f_max - f_min + alpha * total_variation + beta * curvature_penalty
prob = cp.Problem(cp.Minimize(objective), constraints)
prob.solve(solver=cp.SCS)
'''


if f.value is not None:
    import matplotlib.pyplot as plt
    plt.plot(u, f.value, label="f")
    plt.plot(u, w, label="w")
    plt.plot(u, w * f.value, label="f * w")
    plt.title("Voter distribution f(u)")
    plt.xlabel("u")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()

    fval = f.value
    full_A_val = np.sum((u < 0.5) * fval) * du
    weighted_B_val = np.sum((u >= 0.5) * w * fval) * du
    weighted_A_val = np.sum((u < 0.5) * w * fval) * du
    print(f"A share of full turnout: {full_A_val:.4f}")
    print(f"Weighted votes for A: {weighted_A_val:.4f}, B: {weighted_B_val:.4f}")
    print(f"Margin (B - A): {weighted_B_val - weighted_A_val:.4e}")
    
    plt.show()
else:
    print("No feasible solution found.")

