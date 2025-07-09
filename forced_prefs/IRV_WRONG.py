import cvxpy as cp
import numpy as np

def enforce_unequal(x, y, epsilon, name_prefix, M=1.0):
    b = cp.Variable(boolean=True, name=f"{name_prefix}_flip")
    c1 = x - y >= epsilon - M * b
    c2 = y - x >= epsilon - M * (1 - b)
    return [c1, c2]

def solve_irv_difference():
    variables = {}
    var_names = [
        'z_zco', 'z_zoc', 'zo_zoc', 'zc_zco', 'zco_zco', 'zoc_zoc',
        'c_czo', 'c_coz', 'co_coz', 'cz_czo', 'czo_czo', 'coz_coz',
        'o_ozc', 'o_ocz', 'oc_ocz', 'oz_ozc', 'ozc_ozc', 'ocz_ocz'
    ]
    for name in var_names:
        variables[name] = cp.Variable(nonneg=True, name=name)
    P = list(variables.values())

    elim_z = cp.Variable(boolean=True, name='elim_z')
    elim_c = cp.Variable(boolean=True, name='elim_c') 
    elim_o = cp.Variable(boolean=True, name='elim_o')
    constraints = []
    constraints.append(sum(P) == 1)
    constraints.append(elim_z + elim_c + elim_o == 1)
    
    M = 1.0
    epsilon = 1e-7
    
    #first choice tallies
    fc_z = variables['z_zco'] + variables['z_zoc'] + variables['zo_zoc'] + variables['zc_zco'] + variables['zco_zco'] + variables['zoc_zoc']
    fc_c = variables['c_czo'] + variables['c_coz'] + variables['co_coz'] + variables['cz_czo'] + variables['czo_czo'] + variables['coz_coz']
    fc_o = variables['o_ozc'] + variables['o_ocz'] + variables['oc_ocz'] + variables['oz_ozc'] + variables['ozc_ozc'] + variables['ocz_ocz']

    #enforcing o is first eliminated
    constraints += [
        fc_c >= fc_o + epsilon,
        fc_z >= fc_o + epsilon,
        elim_o == 1
    ]
    
    
    constraints += [
        #if elim_z == 1 then fc_z ≤ fc_c and fc_z ≤ fc_o  
        fc_z - fc_c <= M * (1 - elim_z),
        fc_z - fc_o <= M * (1 - elim_z),
        #if elim_c == 1 then fc_c ≤ fc_z and fc_c ≤ fc_o  
        fc_c - fc_z <= M * (1 - elim_c),
        fc_c - fc_o <= M * (1 - elim_c),
        #if elim_o == 1 then fc_o ≤ fc_z and fc_o ≤ fc_c
        fc_o - fc_z <= M * (1 - elim_o),
        fc_o - fc_c <= M * (1 - elim_o)
    ]

    constraints += enforce_unequal(fc_z, fc_c, epsilon, "fc_zc")
    constraints += enforce_unequal(fc_z, fc_o, epsilon, "fc_zo")
    constraints += enforce_unequal(fc_c, fc_o, epsilon, "fc_co")
    
    
    forced_transfers = {
        'z_to_c': variables['z_zco'] + variables['zc_zco'] + variables['zco_zco'],
        'z_to_o': variables['z_zoc'] + variables['zo_zoc'] + variables['zoc_zoc'],
        'c_to_z': variables['c_czo'] + variables['cz_czo'] + variables['czo_czo'],
        'c_to_o': variables['c_coz'] + variables['co_coz'] + variables['coz_coz'],
        'o_to_z': variables['o_ozc'] + variables['oz_ozc'] + variables['ozc_ozc'],
        'o_to_c': variables['o_ocz'] + variables['oc_ocz'] + variables['ocz_ocz']
    }

    partial_transfers = {
        'z_to_c': variables['zc_zco'] + variables['zco_zco'],
        'z_to_o': variables['zo_zoc'] + variables['zoc_zoc'],
        'c_to_z': variables['cz_czo'] + variables['czo_czo'],
        'c_to_o': variables['co_coz'] + variables['coz_coz'],
        'o_to_z': variables['oz_ozc'] + variables['ozc_ozc'],
        'o_to_c': variables['oc_ocz'] + variables['ocz_ocz']
    }

    
    def conditional_transfer(condition, transfer_amount, M):
        aux = cp.Variable(nonneg=True)
        return aux, [
            aux <= M * condition,
            aux <= transfer_amount,
            aux >= transfer_amount - M * (1 - condition),
            aux >= 0
        ]
    
    #tallying forced full prefs
    forced_z_from_c, c1 = conditional_transfer(elim_c, forced_transfers['c_to_z'], M)
    forced_z_from_o, c2 = conditional_transfer(elim_o, forced_transfers['o_to_z'], M)
    forced_c_from_z, c3 = conditional_transfer(elim_z, forced_transfers['z_to_c'], M)
    forced_c_from_o, c4 = conditional_transfer(elim_o, forced_transfers['o_to_c'], M)
    forced_o_from_z, c5 = conditional_transfer(elim_z, forced_transfers['z_to_o'], M)
    forced_o_from_c, c6 = conditional_transfer(elim_c, forced_transfers['c_to_o'], M)
    
    constraints.extend(c1 + c2 + c3 + c4 + c5 + c6)
    z_final_forced = fc_z + forced_z_from_c + forced_z_from_o
    c_final_forced = fc_c + forced_c_from_z + forced_c_from_o
    o_final_forced = fc_o + forced_o_from_z + forced_o_from_c
    
    #enforce c strictly wins under forced
    constraints.extend([
        c_final_forced >= z_final_forced + epsilon,
        c_final_forced >= o_final_forced + epsilon,
        z_final_forced + c_final_forced + o_final_forced <= 1
    ])
    '''
    #tallying partial prefs (O only edition)
    partial_z_from_o, c1 = conditional_transfer(elim_o, partial_transfers['o_to_z'], M)
    partial_c_from_o, c2 = conditional_transfer(elim_o, partial_transfers['o_to_c'], M)
    constraints.extend(c1 + c2)

    z_final_partial = fc_z + partial_z_from_o
    c_final_partial = fc_c + partial_c_from_o
    o_final_partial = fc_o  # o is eliminated

    # enforce z strictly wins with partial
    constraints.extend([
        z_final_partial >= c_final_partial + epsilon,
        z_final_partial >= o_final_partial + epsilon
        #z_final_partial + c_final_partial + o_final_partial <= 1
    ])
    '''
    #tallying partial prefs (more general IRV)
    partial_z_from_c, c1 = conditional_transfer(elim_c, partial_transfers['c_to_z'], M)
    partial_z_from_o, c2 = conditional_transfer(elim_o, partial_transfers['o_to_z'], M)
    partial_c_from_z, c3 = conditional_transfer(elim_z, partial_transfers['z_to_c'], M)
    partial_c_from_o, c4 = conditional_transfer(elim_o, partial_transfers['o_to_c'], M)
    partial_o_from_z, c5 = conditional_transfer(elim_z, partial_transfers['z_to_o'], M)
    partial_o_from_c, c6 = conditional_transfer(elim_c, partial_transfers['c_to_o'], M)

    constraints.extend(c1 + c2 + c3 + c4 + c5 + c6)

    z_final_partial = fc_z + partial_z_from_c + partial_z_from_o
    c_final_partial = fc_c + partial_c_from_z + partial_c_from_o
    o_final_partial = fc_o + partial_o_from_z + partial_o_from_c
    
    #enforce z strictly wins with partial
    constraints.extend([
        z_final_partial >= c_final_partial + epsilon,
        z_final_partial >= o_final_partial + epsilon
        #z_final_partial + c_final_partial + o_final_partial <= 1
    ])
    

    
    #objective = cp.Maximize(c_round2 - c_round2_p)
    objective = cp.Minimize(0)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI, reoptimize=True)
        
    return prob, variables, (z_final_forced, c_final_forced, o_final_forced), (z_final_partial, c_final_partial, o_final_partial)

prob, variables, forced_results, partial_results = solve_irv_difference()

if prob.status == cp.OPTIMAL:
    print("Solution found!")
    print("\nVote distribution:")
    for name, var in variables.items():
        if var.value and var.value > 1e-6:
            print(f"  {name}: {var.value:.8f}")
    
    print(f"\nForced full preferences - Final votes:")
    print(f"  z: {forced_results[0].value:.8f}")
    print(f"  c: {forced_results[1].value:.8f}")
    print(f"  o: {forced_results[2].value:.8f}")
    
    print(f"\nPartial preferences - Votes:")
    print(f"  z: {partial_results[0].value:.8f}")
    print(f"  c: {partial_results[1].value:.8f}")
    print(f"  o: {partial_results[2].value:.8f}")
else:
    print(f"No solution found. Status: {prob.status}")
