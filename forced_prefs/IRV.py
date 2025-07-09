import cvxpy as cp
import inspect

def retrieve_name(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if names:
            return names[0]

def bigM_product(var_bool, expr, M, name=""):
    aux = cp.Variable(name=name)
    constraints = [
        aux <= M * var_bool,
        aux <= expr,
        aux >= expr - M * (1 - var_bool),
        aux >= 0
    ]
    return aux, constraints

def enforce_unequal(x, y, epsilon, name_prefix, M=1.0):
    b = cp.Variable(boolean=True, name=f"{name_prefix}_flip")
    c1 = x - y >= epsilon - M * b
    c2 = y - x >= epsilon - M * (1 - b)
    return [c1, c2]

z_zco = cp.Variable(nonneg=True)
z_zoc = cp.Variable(nonneg=True)
zo_zoc = cp.Variable(nonneg=True)
zc_zco = cp.Variable(nonneg=True)
zco_zco = cp.Variable(nonneg=True)
zoc_zoc = cp.Variable(nonneg=True)

c_czo = cp.Variable(nonneg=True)
c_coz = cp.Variable(nonneg=True)
co_coz = cp.Variable(nonneg=True)
cz_czo = cp.Variable(nonneg=True)
czo_czo = cp.Variable(nonneg=True)
coz_coz = cp.Variable(nonneg=True)

o_ozc = cp.Variable(nonneg=True)
o_ocz = cp.Variable(nonneg=True)
oc_ocz = cp.Variable(nonneg=True)
oz_ozc = cp.Variable(nonneg=True)
ozc_ozc = cp.Variable(nonneg=True)
ocz_ocz = cp.Variable(nonneg=True)

P = [z_zco, z_zoc, zo_zoc, zc_zco, zco_zco, zoc_zoc,
     c_czo, c_coz, co_coz, cz_czo, czo_czo, coz_coz,
     o_ozc, o_ocz, oc_ocz, oz_ozc, ozc_ozc, ocz_ocz]

elim_z, elim_c, elim_o = cp.Variable(boolean=True), cp.Variable(boolean=True), cp.Variable(boolean=True)

M = 1.0
epsilon = 1e-5
constraints = [sum(P) == 1,
               elim_z + elim_c + elim_o == 1]

fc_z = z_zco + z_zoc + zo_zoc + zc_zco + zco_zco + zoc_zoc
fc_c = c_czo + c_coz + co_coz + cz_czo + czo_czo + coz_coz
fc_o = o_ozc + o_ocz + oc_ocz + oz_ozc + ozc_ozc + ocz_ocz

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

forced_z_to_c = z_zco + zc_zco + zco_zco
forced_z_to_o = z_zoc + zo_zoc + zoc_zoc
forced_c_to_z = c_czo + cz_czo + czo_czo
forced_c_to_o = c_coz + co_coz + coz_coz
forced_o_to_z = o_ozc + oz_ozc + ozc_ozc
forced_o_to_c = o_ocz + oc_ocz + ocz_ocz

z_elim_c, c1 = bigM_product(elim_c, forced_c_to_z, M, "z_elim_c")
z_elim_o, c2 = bigM_product(elim_o, forced_o_to_z, M, "z_elim_o")
c_elim_z, c3 = bigM_product(elim_z, forced_z_to_c, M, "c_elim_z")
c_elim_o, c4 = bigM_product(elim_o, forced_o_to_c, M, "c_elim_o")
o_elim_z, c5 = bigM_product(elim_z, forced_z_to_o, M, "o_elim_z")
o_elim_c, c6 = bigM_product(elim_c, forced_o_to_c, M, "o_elim_c")
constraints += c1 + c2 + c3 + c4 + c5 + c6

z_round2 = fc_z + z_elim_c + z_elim_o
c_round2 = fc_c + c_elim_z + c_elim_o
o_round2 = fc_o + o_elim_z + o_elim_c

#enforcing c strictly wins under forced
constraints += [
    c_round2 >= z_round2 + epsilon,
    c_round2 >= o_round2 + epsilon
]

#enforcing o is first eliminated
constraints += [
    fc_c >= fc_o + epsilon,
    fc_z >= fc_o + epsilon
]

constraints += enforce_unequal(fc_z, fc_c, epsilon, "fc_zc_p")
constraints += enforce_unequal(fc_z, fc_o, epsilon, "fc_zo_p")
constraints += enforce_unequal(fc_c, fc_o, epsilon, "fc_co_p")

partial_z_to_c = zc_zco + zco_zco
partial_o_to_c = oc_ocz + ocz_ocz

c_partial_z, c7 = bigM_product(elim_z, partial_z_to_c, M, "c_partial_z")
c_partial_o, c8 = bigM_product(elim_o, partial_o_to_c, M, "c_partial_o")

constraints += c7 + c8

c_round2_p = fc_c + c_partial_z + c_partial_o

# enforce z strictly wins with partial prefs
constraints += [
    z_round2 >= c_round2_p + epsilon,
    z_round2 >= o_round2 + epsilon
]
#objective = cp.Maximize(c_round2 - c_round2_p)
objective = cp.Minimize(0)
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.GUROBI, reoptimize=True)

if prob.status == cp.OPTIMAL:
    print("Found valid distribution.")
    print("c wins in forced full IRV, z wins in partial IRV.")
    print("Vote distribution:")
    for var in P:
        if var.value and var.value > 1e-6:
            print(f"  {retrieve_name(var)}: {var.value:.6f}")

    print(f"c_round2 (forced): {c_round2.value:.6f}")
    print(f"c_round2 (partial): {c_round2_p.value:.6f}")
    print(f"z_round2 (partial): {z_round2.value:.6f}")
    exhausted_partial = 1 - (z_round2.value + c_round2_p.value)
    print(f"Exhausted ballots in partial IRV: {exhausted_partial:.6f}")
    
else:
    print(f"No solution found. Status: {prob.status}")


