
import inspect
import cvxpy as cp

def retrieve_name(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

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
constraints = [z_zco + z_zoc + zo_zoc + zc_zco + zco_zco + zoc_zoc + c_czo + c_coz + co_coz + cz_czo + czo_czo + coz_coz + o_ozc + o_ocz + oc_ocz + oz_ozc + ozc_ozc + ocz_ocz == 1]

def borda_forced_full():
    z = 2 * (z_zco + z_zoc + zo_zoc + zc_zco + zco_zco + zoc_zoc) + 1 * ((c_czo + cz_czo + czo_czo) + (o_ozc + oz_ozc + ozc_ozc))
    c = 2 * (c_czo + c_coz + co_coz + cz_czo + czo_czo + coz_coz) + 1 * ((z_zco + zc_zco + zco_zco) + (o_ocz + oc_ocz + ocz_ocz))
    o = 2 * (o_ozc + o_ocz + oc_ocz + oz_ozc + ozc_ozc + ocz_ocz) + 1 * ((z_zoc + zo_zoc + zoc_zoc) + (c_coz + co_coz + coz_coz))
    return z, c, o

def borda_partial():
    z = 2 * (z_zco + z_zoc + zo_zoc + zc_zco + zco_zco + zoc_zoc) + 1 * ((cz_czo + czo_czo) + (oz_ozc + ozc_ozc))
    c = 2 * (c_czo + c_coz + co_coz + cz_czo + czo_czo + coz_coz) + 1 * ((zc_zco + zco_zco) + (oc_ocz + ocz_ocz))
    o = 2 * (o_ozc + o_ocz + oc_ocz + oz_ozc + ozc_ozc + ocz_ocz) + 1 * ((zo_zoc + zoc_zoc) + (co_coz + coz_coz))
    return z, c, o

z_p, c_p, o_p = borda_partial()
z_f, c_f, o_f = borda_forced_full()

constraints += [
    z_p >= c_p,
    z_p >= o_p,
    c_f >= z_f + 1e-4,  #epsilon to force strictness?
    c_f >= o_f + 1e-4,
]

prob = cp.Problem(cp.Minimize(0), constraints)
prob.solve()

if prob.status == cp.OPTIMAL:
    print("Found distribution where:")
    print("  Borda winner with partial ballots is z")
    print("  Borda winner with forced full ballots is c")
    for var in P:
        if var.value > 1e-5:
            print(f"  {retrieve_name(var)}: {var.value:.4f}")
else:
    print("No such distribution found.")

