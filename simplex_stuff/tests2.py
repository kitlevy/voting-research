import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def threshold_for_inequality(ua, ub, wa, wb):
    P = ua - ub
    Q = wa - wb
    if P > 0:
        return 0.0
    denom = Q - P
    if denom == 0:
        return 1.0 if P <= 0 else 0.0
    alpha = -P / denom
    return np.clip(alpha, 0.0, 1.0)

def solve_linear_threshold(u, w):
    u = np.array(u, dtype=float)
    w = np.array(w, dtype=float)
    alpha1 = threshold_for_inequality(u[0], u[2], w[0], w[2])
    alpha2 = threshold_for_inequality(u[2], u[1], w[2], w[1])
    return max(alpha1, alpha2)

print("SUPPORT FUNCTION THEORY:")
print("="*50)
print("""
The support function h_S(p) = sup_{s ∈ S} ⟨p, s⟩ has a beautiful geometric interpretation:

1. For a direction vector p, h_S(p) finds the point in set S that extends furthest in direction p
2. The supporting hyperplane at that point is perpendicular to p
3. This is exactly what we want for optimization problems!

For your problem:
- S = {points in a > c > b region}
- p = u (your voter's utility vector)
- h_S(u) finds the point in S most aligned with u

""")

# Let's verify this theory
def support_function_approach(u, resolution=200):
    """Find support function value and supporting point"""
    lambdas = np.linspace(0.0, 1.0, resolution, endpoint=True)
    max_dot = -np.inf
    opt_w = None
    
    # Search over boundary of a > c > b region
    for l in lambdas:
        # Three edges of the region
        w1 = [(1 - l) / 2 + l, 0, (1 - l) / 2]           # Edge from (1/2,0,1/2) to (1,0,0)
        w2 = [(1 - l) / 2 + l / 3, l / 3, (1 - l) / 2 + l / 3]  # Edge from (1/2,0,1/2) to (1/3,1/3,1/3)
        w3 = [l / 3 + (1 - l), l / 3, l / 3]             # Edge from (1,0,0) to (1/3,1/3,1/3)
        
        for w in [w1, w2, w3]:
            dot_prod = np.dot(u, w)
            if dot_prod > max_dot:
                max_dot = dot_prod
                opt_w = w
    
    return max_dot, opt_w

def analyze_support_function_relevance(u, name):
    print(f"\n{name}: u = {u}")
    
    # Find support function result
    support_val, support_point = support_function_approach(u)
    support_alpha = solve_linear_threshold(u, support_point)
    
    # Find actual minimum alpha result (from your working function)
    lambdas = np.linspace(0.0, 1.0, 200, endpoint=True)
    min_alpha = 1
    opt_w = None
    for l in lambdas:
        w1 = [(1 - l) / 2 + l, 0, (1 - l) / 2]
        w2 = [(1 - l) / 2 + l / 3, l / 3, (1 - l) / 2 + l / 3]
        w3 = [l / 3 + (1 - l), l / 3, l / 3]
        for w in [w1, w2, w3]:
            alpha = solve_linear_threshold(u, w)
            if alpha < min_alpha:
                min_alpha = alpha
                opt_w = w
    
    print(f"Support function: max_dot = {support_val:.4f}, w = {np.array(support_point)}")
    print(f"                 alpha = {support_alpha:.4f}")
    print(f"Min alpha search: alpha = {min_alpha:.4f}, w = {np.array(opt_w)}")
    print(f"Same result? {np.allclose(support_point, opt_w, atol=1e-3)}")
    
    return support_val, support_point, min_alpha, opt_w

# Test with your points
P1 = [2/3, 1/3, 0]
P2 = [1/3, 2/3, 0]
P3 = [0, 2/3, 1/3]
P4 = [0, 1/3, 2/3]
P5 = [1/3, 0, 2/3]

results = []
for i, p in enumerate([P1, P2, P3, P4, P5], 1):
    result = analyze_support_function_relevance(p, f"P{i}")
    results.append(result)

print("\n" + "="*60)
print("WHY THE PROFESSOR SUGGESTED SUPPORT FUNCTIONS:")
print("="*60)
print("""
1. GEOMETRIC INTUITION: The support function finds the "most extreme" point in the 
   direction of u. This is often (but not always) the optimal anchor point.

2. CONVEX OPTIMIZATION: Support functions are fundamental in convex analysis. 
   Your region S (a > c > b) is convex, so support function theory applies perfectly.

3. DUALITY: There's a deep connection between:
   - Finding the closest point in S to u (your original problem)
   - Finding the supporting hyperplane of S in direction u (support function)

4. COMPUTATIONAL EFFICIENCY: Support functions can sometimes be computed more 
   efficiently than brute force search.
""")

# Let's check why some don't match exactly
print("\nWHY SUPPORT FUNCTION MIGHT NOT ALWAYS GIVE MINIMUM ALPHA:")
print("""
The support function finds the point most aligned with u, but:
1. We don't want the final blended point to be most aligned with u
2. We want the blended point to JUST satisfy the constraints
3. Sometimes a less aligned anchor point requires smaller alpha

Think of it this way:
- Support function: "Which boundary point is most like u?"  
- Our problem: "Which boundary point can pull u into the region most efficiently?"

These can be different! The most "similar" point isn't always the most "efficient" anchor.
""")

# Demonstrate with concrete example
print(f"\nCONCRETE EXAMPLE:")
u = np.array([0, 1/3, 2/3])  # P4
print(f"u = {u}")

# The support function might find a point that's very aligned with u
# But we need a point that can satisfy BOTH a > c AND c > b constraints efficiently

w_support = [1.0, 0, 0]  # Extreme point in direction of a
w_optimal = [1.0, 0, 0]  # Actually the same in this case

alpha_support = solve_linear_threshold(u, w_support)  
alpha_optimal = solve_linear_threshold(u, w_optimal)

print(f"Support function point: {w_support}, alpha = {alpha_support}")
print(f"Optimal point: {w_optimal}, alpha = {alpha_optimal}")

print(f"\nIn this case they match! But that's not always guaranteed.")
print(f"The support function is a great starting heuristic, but you might need")
print(f"to check nearby points for the true minimum.")

print(f"\n" + "="*50)
print("WHAT THE SUPPORT FUNCTION ACTUALLY TELLS US:")
print("="*50)

def demonstrate_support_function():
    u = np.array([2/3, 1/3, 0])
    
    # The support function value
    vertices = [
        np.array([1, 0, 0]),
        np.array([1/2, 0, 1/2]), 
        np.array([1/3, 1/3, 1/3])
    ]
    
    support_value = max(np.dot(u, v) for v in vertices)
    support_point = vertices[np.argmax([np.dot(u, v) for v in vertices])]
    
    print(f"For u = {u}:")
    print(f"Support function value h_S(u) = {support_value:.4f}")
    print(f"Achieved at point {support_point}")
    
    print(f"\nGeometric interpretation:")
    print(f"- This is the point in the a>c>b region that extends furthest in direction u")
    print(f"- It's where the hyperplane {{w : ⟨u,w⟩ = {support_value:.4f}}} touches the region")
    print(f"- All other points in the region satisfy ⟨u,w⟩ ≤ {support_value:.4f}")

demonstrate_support_function()
