import numpy as np
from scipy.stats import beta
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def beta_pdf(u, alpha_, beta_):
    return beta.pdf(u, alpha_, beta_)

# Constraint 1: F(1/2) >= 1/2
def constraint_majority_prefers_a(alpha_beta):
    alpha_, beta_ = alpha_beta
    try:
        incomplete_beta = beta.cdf(0.5, alpha_, beta_)
        result = incomplete_beta - 0.5 - epsilon
        print(f"  Constraint 1 at ({alpha_:.3f}, {beta_:.3f}): CDF(0.5)={incomplete_beta:.6f}, constraint={result:.6f}")
        return result
    except Exception as e:
        print(f"  Error in constraint 1: {e}")
        return -1e6

# Constraint 2: expected votes for A <= expected votes for B
def expected_votes(alpha_, beta_):
    def vote_a(u):
        return (1 - 2 * u) * beta_pdf(u, alpha_, beta_)
    
    def vote_b(u):
        return (2 * u - 1) * beta_pdf(u, alpha_, beta_)
    
    try:
        va, _ = quad(vote_a, 0, 0.5)
        vb, _ = quad(vote_b, 0.5, 1)
        return va, vb
    except Exception as e:
        print(f"  Error in expected_votes: {e}")
        return 0, 0

def constraint_expected_votes(alpha_beta):
    alpha_, beta_ = alpha_beta
    try:
        va, vb = expected_votes(alpha_, beta_)
        result = vb - va - epsilon
        print(f"  Constraint 2 at ({alpha_:.3f}, {beta_:.3f}): VA={va:.6f}, VB={vb:.6f}, constraint={result:.6f}")
        return result
    except Exception as e:
        print(f"  Error in constraint 2: {e}")
        return -1e6

# Test if constraints are satisfiable
def test_constraint_feasibility():
    print("=== Testing Constraint Feasibility ===")
    
    # Test a range of alpha, beta values
    alphas = np.linspace(0.1, 5, 10)
    betas = np.linspace(0.1, 5, 10)
    
    feasible_points = []
    
    for alpha_val in alphas:
        for beta_val in betas:
            point = [alpha_val, beta_val]
            c1 = constraint_majority_prefers_a(point)
            c2 = constraint_expected_votes(point)
            
            if c1 >= 0 and c2 >= 0:
                feasible_points.append((alpha_val, beta_val, c1, c2))
                print(f"FEASIBLE: α={alpha_val:.2f}, β={beta_val:.2f}, C1={c1:.6f}, C2={c2:.6f}")
    
    print(f"\nFound {len(feasible_points)} feasible points out of {len(alphas)*len(betas)} tested")
    return feasible_points

# Objective functions
def objective_maximize_gap(alpha_beta):
    alpha_, beta_ = alpha_beta
    return -abs(alpha_ - beta_)

def objective_with_debug(alpha_beta):
    alpha_, beta_ = alpha_beta
    obj_val = objective_maximize_gap(alpha_beta)
    print(f"  Objective at ({alpha_:.3f}, {beta_:.3f}): {obj_val:.6f}")
    return obj_val

if __name__ == "__main__":
    epsilon = 1e-6
    
    # First, test constraint feasibility
    feasible_points = test_constraint_feasibility()
    
    if not feasible_points:
        print("\n❌ No feasible points found! Constraints are incompatible.")
        
        # Let's analyze why - create a heatmap
        print("\n=== Constraint Analysis ===")
        alphas = np.linspace(0.1, 3, 20)
        betas = np.linspace(0.1, 3, 20)
        
        c1_values = np.zeros((len(alphas), len(betas)))
        c2_values = np.zeros((len(alphas), len(betas)))
        
        for i, alpha_val in enumerate(alphas):
            for j, beta_val in enumerate(betas):
                # Temporarily disable printing for heatmap
                old_constraint_1 = constraint_majority_prefers_a
                old_constraint_2 = constraint_expected_votes
                
                def silent_constraint_1(ab):
                    alpha_, beta_ = ab
                    incomplete_beta = beta.cdf(0.5, alpha_, beta_)
                    return incomplete_beta - 0.5 - epsilon
                
                def silent_constraint_2(ab):
                    alpha_, beta_ = ab
                    va, vb = expected_votes(alpha_, beta_)
                    return vb - va - epsilon
                
                c1_values[i, j] = silent_constraint_1([alpha_val, beta_val])
                c2_values[i, j] = silent_constraint_2([alpha_val, beta_val])
        
        # Find regions where each constraint is satisfied
        c1_satisfied = c1_values >= 0
        c2_satisfied = c2_values >= 0
        both_satisfied = c1_satisfied & c2_satisfied
        
        print(f"Points where C1 ≥ 0: {np.sum(c1_satisfied)}/{c1_satisfied.size}")
        print(f"Points where C2 ≥ 0: {np.sum(c2_satisfied)}/{c2_satisfied.size}")
        print(f"Points where both ≥ 0: {np.sum(both_satisfied)}/{both_satisfied.size}")
        
        # Find the closest points to feasibility
        constraint_violations = np.minimum(c1_values, 0) + np.minimum(c2_values, 0)
        best_idx = np.unravel_index(np.argmax(constraint_violations), constraint_violations.shape)
        best_alpha, best_beta = alphas[best_idx[0]], betas[best_idx[1]]
        
        print(f"\nClosest to feasible: α={best_alpha:.3f}, β={best_beta:.3f}")
        print(f"  C1 = {c1_values[best_idx]:.6f}")
        print(f"  C2 = {c2_values[best_idx]:.6f}")
        
    else:
        print(f"\n✅ Found {len(feasible_points)} feasible points. Trying optimization...")
        
        # Use the first feasible point as initial guess
        initial_guess = [feasible_points[0][0], feasible_points[0][1]]
        print(f"Starting from feasible point: α={initial_guess[0]:.3f}, β={initial_guess[1]:.3f}")
        
        bounds = [(0.1, 10.0), (0.1, 10.0)]
        constraints = [
            {'type': 'ineq', 'fun': constraint_majority_prefers_a},
            {'type': 'ineq', 'fun': constraint_expected_votes}
        ]
        
        # Try different optimization methods
        methods = ['SLSQP', 'COBYLA', 'trust-constr']
        
        for method in methods:
            print(f"\n--- Trying {method} method ---")
            
            try:
                if method == 'COBYLA':
                    # COBYLA doesn't support bounds, only constraints
                    bounds_constraints = [
                        {'type': 'ineq', 'fun': lambda x: x[0] - 0.1},
                        {'type': 'ineq', 'fun': lambda x: 10.0 - x[0]},
                        {'type': 'ineq', 'fun': lambda x: x[1] - 0.1},
                        {'type': 'ineq', 'fun': lambda x: 10.0 - x[1]}
                    ]
                    all_constraints = constraints + bounds_constraints
                    result = minimize(objective_with_debug, initial_guess, 
                                    method=method, constraints=all_constraints, 
                                    options={'disp': True, 'maxiter': 1000})
                else:
                    result = minimize(objective_with_debug, initial_guess, 
                                    method=method, bounds=bounds, constraints=constraints, 
                                    options={'disp': True, 'maxiter': 1000})
                
                print(f"Result: {result.message}")
                
                if result.success:
                    alpha_opt, beta_opt = result.x
                    print(f"Optimal values: α={alpha_opt:.6f}, β={beta_opt:.6f}")
                    va, vb = expected_votes(alpha_opt, beta_opt)
                    a_share = beta.cdf(0.5, alpha_opt, beta_opt)
                    print(f"Expected votes: A={va:.6f}, B={vb:.6f}")
                    print(f"Share preferring A: {a_share:.6f}")
                    break
                    
            except Exception as e:
                print(f"Error with {method}: {e}")
        
        if not any(method == 'SLSQP' for method in methods):
            print("\n❌ No method succeeded")
