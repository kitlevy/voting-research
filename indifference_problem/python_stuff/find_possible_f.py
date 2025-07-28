import cvxpy as cp
import numpy as np

N = 500  
u = np.linspace(0, 1, N)
du = u[1] - u[0]

w = np.abs(2 * u - 1)

f = cp.Variable(N)

weighted_B = cp.sum(cp.multiply(w * (u >= 0.5), f)) * du
weighted_A = cp.sum(cp.multiply(w * (u < 0.5), f)) * du
full_A = cp.sum(cp.multiply((u < 0.5), f)) * du

constraints = [
    cp.sum(f) * du == 1,              #normalize f to integrate to 1
    full_A >= 0.5 + 1e-3,             #A wins in full turnout
    weighted_B >= weighted_A + 1e-3,  #B wins in weighted turnout
    f >= 0,                           #non-negative density
]

#Lipschitz constraint (approximate)
L = 50  #max derivative
for i in range(N - 1):
    constraints.append(cp.abs(f[i + 1] - f[i]) <= L * du)

prob = cp.Problem(cp.Minimize(0), constraints)
prob.solve()

if f.value is not None:
    import matplotlib.pyplot as plt
    plt.plot(u, f.value)
    plt.title("Voter distribution f(u)")
    plt.xlabel("u")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()
else:
    print("No feasible solution found.")

