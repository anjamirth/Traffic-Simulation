import math
from cars_NaSch_plot_updated import run_sim as run_sim_cars
from run_experiment_plot_data import run_sim as run_sim_dict

def first_mismatch(a, b, name):
    # a, b are lists
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, a[i], b[i], name
    if len(a) != len(b):
        return n, "(end)", "(end)", f"{name} (length differs)"
    return None

def compare_series(series_a, series_b, keys):
    for k in keys:
        mm = first_mismatch(series_a[k], series_b[k], k)
        if mm is not None:
            i, va, vb, name = mm
            print(f"\n❌ MISMATCH in '{name}' at t={i}")
            print(f"dict={va}")
            print(f"cars={vb}")

            # print local window around mismatch
            lo = max(0, i-5)
            hi = min(len(series_a[k]), i+6)
            print(f"\nWindow t={lo}..{hi-1} for '{k}':")
            print("dict:", series_a[k][lo:hi])
            print("cars:", series_b[k][lo:hi])
            return False
    print("\n✅ All compared series match exactly.")
    return True

# ----------------------------
# YOU set these:
# ----------------------------
penetration_rate = 0.50
seed = 1

# ----------------------------
# YOU must provide these two functions (import them if in separate files):
#   meta, series = run_sim_dict(penetration_rate, seed)
#   meta, series = run_sim_cars(penetration_rate, seed)
# ----------------------------

meta1, series1 = run_sim_dict(auto_frac=penetration_rate, seed=seed)
meta2, series2 = run_sim_cars(penetration_rate=penetration_rate, seed=seed)

print("meta_dict:", meta1)
print("meta_cars:", meta2)

keys_to_check = ["moved", "entered", "queued", "jammed_cells", "jam_clusters"]
ok = compare_series(series1, series2, keys_to_check)
