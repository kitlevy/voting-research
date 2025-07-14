import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def find_example(epsilon=1e-5):
    z_zco = cp.Variable(nonneg=True)
    z_zoc = cp.Variable(nonneg=True)
    oz_ozc = cp.Variable(nonneg=True)
    ozc_ozc = cp.Variable(nonneg=True)

    c_czo = cp.Variable(nonneg=True)
    c_coz = cp.Variable(nonneg=True)
    oc_ocz = cp.Variable(nonneg=True)
    ocz_ocz = cp.Variable(nonneg=True)

    o_ozc = cp.Variable(nonneg=True)
    o_ocz = cp.Variable(nonneg=True)

    all_vars = [z_zco, z_zoc, oz_ozc, ozc_ozc,
                c_czo, c_coz, oc_ocz, ocz_ocz,
                o_ozc, o_ocz]

    z_partial = z_zco + z_zoc + oz_ozc + ozc_ozc
    c_partial = c_czo + c_coz + oc_ocz + ocz_ocz

    z_forced = z_partial + o_ozc
    c_forced = c_partial + o_ocz

    constraints = [
        cp.sum(all_vars) == 1,
        z_partial >= c_partial + epsilon,
        c_forced >= z_forced + epsilon,
    ]

    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()

    if prob.status == cp.OPTIMAL:
        return {
            "z_partial": z_partial.value,
            "c_partial": c_partial.value,
            "z_forced": z_forced.value,
            "c_forced": c_forced.value,
            "exhausted_partial": 1 - (z_partial.value + c_partial.value),
            "vars": {v.name(): v.value for v in all_vars if v.value > 1e-6}
        }
    else:
        return None

def sweep_plot():
    oz_vals = np.linspace(0, 0.5, 50)
    oc_vals = np.linspace(0, 0.5, 50)
    win_diff = np.full((len(oz_vals), len(oc_vals)), np.nan)

    for i, oz in enumerate(oz_vals):
        for j, oc in enumerate(oc_vals):
            z_zco = 0.3
            z_zoc = 0.2
            c_czo = 0.2
            c_coz = 0.1
            oz_ozc = oz
            oc_ocz = oc
            remaining = 1 - (z_zco + z_zoc + c_czo + c_coz + oz + oc)
            if remaining < 0:
                continue
            ozc_ozc = remaining / 2
            ocz_ocz = remaining / 2

            z_partial = z_zco + z_zoc + oz + ozc_ozc
            c_partial = c_czo + c_coz + oc + ocz_ocz
            z_forced = z_partial + 0  # no o->z beyond oz
            c_forced = c_partial + 0  # no o->c beyond oc

            win_diff[i, j] = (z_partial - c_partial) - (z_forced - c_forced)

    plt.figure(figsize=(8, 6))
    plt.contourf(oc_vals, oz_vals, win_diff, levels=20, cmap="coolwarm", alpha=0.8)
    plt.colorbar(label="(z - c) partial minus (z - c) forced")
    plt.xlabel("oc_ocz (o → c in partial)")
    plt.ylabel("oz_ozc (o → z in partial)")
    plt.title("Flip Region: z wins partial, c wins forced")
    plt.axhline(0.25, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0.25, color='gray', linestyle='--', linewidth=0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

example = find_example()
if example:
    print("Found valid reversal example:")
    print(example)
else:
    print("No valid example found.")

sweep_plot()

