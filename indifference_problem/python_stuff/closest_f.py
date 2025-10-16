import numpy as np
import cvxpy as cp

N = 1000
u = np.linspace(0, 1, N)
du = u[1] - u[0]

w = np.abs(u - 1/2) + 1/2
#w = (u - 1/2) ** 2

w_string = "|u - 1/2| + 1/2"
#w_string = "(u - 1/2)^2"

left = u < 0.5
right = u > 0.5

f = cp.Variable(N, nonneg=True)
J = cp.sum(cp.multiply(w[right], f[right])) * du - cp.sum(cp.multiply(w[left], f[left])) * du
objective = cp.Minimize(J)

constraints = [
    cp.sum(f) * du == 1,  # total probability = 1
    cp.sum(f[left]) * du >= 0.5,  # A wins full turnout
    J >= 0  # B wins weighted turnout
]

problem = cp.Problem(objective, constraints)
problem.solve()

if f.value is not None:
    import matplotlib.pyplot as plt
    plt.plot(u, f.value, label="f")
    plt.plot(u, w, label="w")
    plt.plot(u, w * f.value, label="f * w")
    plt.suptitle("Voter distribution f(u)")
    plt.title("w = " + w_string)
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
