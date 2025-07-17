import numpy as np
from scipy.stats import beta
from scipy.integrate import quad
from scipy.optimize import minimize

def beta_pdf(u, alpha, beta_):
    return beta.pdf(u, alpha, beta_)

#constraint 1: F(1/2) >= 1/2
def constraint_majority_prefers_a(alpha_beta):
    alpha, beta_ = alpha_beta
    #incomplete beta function I(alpha, beta, 1/2)
    incomplete_beta = beta.cdf(0.5, alpha, beta_)
    return 0.5 - epsilon - incomplete_beta  #must be > 0

#constraint 2: expected votes for a ≤ expected votes for b
def expected_votes(alpha, beta_):
    def vote_a(u):
        return (1 - 2 * u) * beta_pdf(u, alpha, beta_)
    def vote_b(u):
        return (2 * u - 1) * beta_pdf(u, alpha, beta_)
    
    va, _ = quad(vote_a, 0, 0.5)
    vb, _ = quad(vote_b, 0.5, 1)
    
    return va, vb

def constraint_expected_votes(alpha_beta):
    alpha, beta_ = alpha_beta
    va, vb = expected_votes(alpha, beta_)
    return vb - va - epsilon

#dummy objective function
def objective(alpha_beta):
    return 0

def objective_maximize_gap(alpha_beta):
    alpha_, beta_ = alpha_beta
    return -abs(alpha_ - beta_)

epsilon = 1e-5

bounds = [(0.1, 10.0), (0.1, 10.0)]
constraints = [ #must be dict for scipy optimization
    {'type': 'ineq', 'fun': constraint_majority_prefers_a},
    {'type': 'ineq', 'fun': constraint_expected_votes}
]
initial_guess = [1, 1]  #start w/ uniform beta distribution
result = minimize(objective_maximize_gap, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP', options={'disp': True, 'maxiter': 1000})

if result.success:
    alpha_opt, beta_opt = result.x
    print(f"Values found: alpha = {alpha_opt:.8f}, beta = {beta_opt:.8f}")
    votes = expected_votes(alpha_opt, beta_opt)
    a_share = beta.cdf(0.5, alpha_opt, beta_opt)
    print(f"Expected votes for a: {votes[0]}, expected votes for b: {votes[1]}")
    print(f"Share of voters who prefer a: {a_share}")
else:
    print("No solution found.")

