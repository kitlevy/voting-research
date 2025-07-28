import numpy as np
from scipy.stats import beta
from scipy.integrate import quad
from scipy.optimize import minimize

epsilon = 1e-8

def turnout_func(u):
    #return np.abs(2 * u - 1)
    return (2 * u - 1) ** 2

def w(u):
    return (2 * u - 1)

#constraint 1: a wins unweighted election
def constraint_majority_prefers_a(var):
    u_a, u_b, p_a = var
    return p_a - (1 - p_a) #must be > 0

#constraint 2: expected votes for a ≤ expected votes for b
def constraint_expected_votes(var):
    u_a, u_b, p_a = var
    ex_a_votes = w(var[0]) * var[2]
    ex_b_votes = w(var[1]) * (1 - var[2])
    return ex_b_votes - ex_a_votes #must be > 0

#dummy objective function
def objective(var):
    u_a, u_b, p_a = var
    return (w(u_a) * p_a - w(u_b) * (1 - p_a))**2  

if __name__ == "__main__":
    bounds = [(0 + epsilon, 0.5 - epsilon), (0.5 + epsilon, 1.0 - epsilon), (0.5 + epsilon, 0.9)]
    constraints = [ #must be dict for scipy optimization
                   {'type': 'ineq', 'fun': lambda var: w(var[0]) * var[2] - w(var[1]) * (1 - var[2]) - epsilon},
                   {'type': 'ineq', 'fun': lambda var: var[2] - (1 - var[2]) - epsilon}
    ]
    initial_guess = [1/3, 2/3, 0.5]
    #initial_guess = [1/3, 1.0 - epsilon, 0.5]
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, options={'disp': True, 'maxiter': 1000, 'tol': 1e-10})
    if result.success:
        u_a, u_b, p_a = result.x
        print("u_a: {}, u_b: {}, p_a: {}".format(u_a, u_b, p_a))
        a_votes = w(u_a) * p_a
        b_votes = w(u_b) * (1 - p_a)
        print(f"Share of voters who prefer a: {p_a}")
        print(f"Expected votes for a: {a_votes}, expected votes for b: {b_votes}")
        print(f"Initial guess: {initial_guess}")
        print("Majority prefers a:", constraint_majority_prefers_a(result.x))
        print("Ex(b) > Ex(a):", constraint_expected_votes(result.x))
    else:
        print("No solution found.")

