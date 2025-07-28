import numpy as np
from scipy.stats import beta
from scipy.integrate import quad
from scipy.optimize import minimize

def beta_pdf(u, alpha_, beta_):
    return beta.pdf(u, alpha_, beta_)

#constraint 1: F(1/2) >= 1/2
def constraint_majority_prefers_a(alpha_beta):
    alpha_, beta_ = alpha_beta
    #incomplete beta function I(alpha_, beta, 1/2)
    incomplete_beta = beta.cdf(0.5, alpha_, beta_)
    return incomplete_beta - 0.5 - epsilon #must be > 0

#constraint 2: expected votes for a ≤ expected votes for b
def expected_votes(alpha_, beta_):
    def vote_a(u):
        if u > 0.5:
            return 0
        return (1 - 2 * u) * beta_pdf(u, alpha_, beta_)
    def vote_b(u):
        if u < 0.5:
            return 0
        return (2 * u - 1) * beta_pdf(u, alpha_, beta_)
    
    va, _ = quad(vote_a, 0, 0.5)
    vb, _ = quad(vote_b, 0.5, 1)
    
    return va, vb

def constraint_expected_votes(alpha_beta):
    alpha_, beta_ = alpha_beta
    va, vb = expected_votes(alpha_, beta_)
    return vb - va - epsilon

#dummy objective function
def objective(alpha_beta):
    return 0

def objective_maximize_gap(alpha_beta):
    alpha_, beta_ = alpha_beta
    return -(alpha_-beta_)**2 

def objective_maximize_distortion(alpha_beta):
    alpha_, beta_ = alpha_beta

    def integrand_total_sw(u):
        return beta_pdf(u, alpha_, beta_)

    def integrand_vote_sw(u):
        turnout_prob = abs(2 * u - 1)
        chosen_util = u if u > 0.5 else 1 - u
        return turnout_prob * beta_pdf(u, alpha_, beta_)

    total_sw, _ = quad(integrand_total_sw, 0, 1)
    voted_sw, _ = quad(integrand_vote_sw, 0, 1)

    if voted_sw < 1e-8:
        return 1e6

    distortion = total_sw / voted_sw
    return -distortion

if __name__ == "__main__":

    epsilon = 0

    bounds = [(0.1, 10.0), (0.1, 10.0)]
    #bounds = [(0.2, 5), (0.2, 5)]
    constraints = [ #must be dict for scipy optimization
        {'type': 'ineq', 'fun': constraint_majority_prefers_a},
        {'type': 'ineq', 'fun': constraint_expected_votes}
    ]
    initial_guess = [2.0, 4.0]
    #initial_guess = [9.9 * np.random.rand() + 0.1, 9.9 * np.random.rand() + 0.1]  #start w/ uniform beta distribution
    #result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, options={'disp': True, 'maxiter': 1000}, method='COBYLA')
    result = minimize(objective_maximize_gap, initial_guess, bounds=bounds, constraints=constraints, options={'disp': True, 'maxiter': 1000}, method='COBYLA')
    #result = minimize(objective_maximize_distortion, initial_guess, bounds=bounds, constraints=constraints, options={'disp': True, 'maxiter': 1000}, method='COBYLA')

    if result.success:
        alpha_opt, beta_opt = result.x
        print(f"alpha = {alpha_opt:.12f}, beta = {beta_opt:.12f}")
        votes = expected_votes(alpha_opt, beta_opt)
        a_share = beta.cdf(0.5, alpha_opt, beta_opt)
        print(f"Expected votes for a: {votes[0]}, expected votes for b: {votes[1]}")
        print(f"Share of voters who prefer a: {a_share}")
        print(f"Initial guess: {initial_guess}")
        print("Majority prefers a:", constraint_majority_prefers_a(result.x))
        print("Ex(b) > Ex(a):", constraint_expected_votes(result.x))
    else:
        print("No solution found.")

