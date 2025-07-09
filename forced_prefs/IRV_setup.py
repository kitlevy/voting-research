import inspect

def retrieve_name(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

import cvxpy as cp

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

P = [z_zco, z_zoc, zo_zoc, zc_zco, zco_zco, zoc_zoc, c_czo, c_coz, co_coz, cz_czo, czo_czo, coz_coz, o_ozc, o_ocz, oc_ocz, oz_ozc, ozc_ozc, ocz_ocz]

elim_z, elim_c, elim_o = cp.Variable(boolean=True), cp.Variable(boolean=True), cp.Variable(boolean=True)
win_z, win_c, win_o = cp.Variable(boolean=True), cp.Variable(boolean=True), cp.Variable(boolean=True)

forced = True
epsilon = 1e-5

constraints = [sum(P) == 1, elim_z + elim_c + elim_o == 1, win_z + win_c + win_o == 1]

fc_z = z_zco + z_zoc + zo_zoc + zc_zco + zco_zco + zoc_zoc
fc_c = c_czo + c_coz + co_coz + cz_czo + czo_czo + coz_coz
fc_o = o_ozc + o_ocz + oc_ocz + oz_ozc + ozc_ozc + ocz_ocz

# IRV elim constraints with big M logic
M = 1.0
constraints += [
    #if elim_z = 1 then fc_z ≤ fc_c and fc_z ≤ fc_o
    fc_z - fc_c <= M * (1 - elim_z),
    fc_z - fc_o <= M * (1 - elim_z),
    #if elim_c = 1 then fc_c ≤ fc_z and fc_c ≤ fc_o
    fc_c - fc_z <= M * (1 - elim_c),
    fc_c - fc_o <= M * (1 - elim_c),
    #if elim_o = 1 then fc_o ≤ fc_z and fc_o ≤ fc_c
    fc_o - fc_z <= M * (1 - elim_o),
    fc_o - fc_c <= M * (1 - elim_o)
]

#rd 2 totals, winner constraints
z_round2 = fc_z + elim_c * (forced * z_zco + zc_zco + zco_zco) + elim_o * (forced * z_zoc + zo_zoc + zoc_zoc)
c_round2 = fc_c + elim_z * (forced * z_zco + zc_zco + zco_zco) + elim_o * (forced * o_ocz + oc_ocz + ocz_ocz)
o_round2 = fc_o + elim_z * (forced * z_zoc + zo_zoc + zoc_zoc) + elim_c * (forced * c_coz + co_coz + coz_coz)

constraints += [
    win_c == 1,
    c_round2 >= z_round2 + epsilon,
    c_round2 >= o_round2 + epsilon,
]
