from Cars import Cars
import random
import numpy as np
import matplotlib.pyplot as plt
import csv

# ----------------------------
# Ring-road parameters (stabilised)
# ----------------------------
L = 100
v_max = 5
num_cars = 20          # was 15; 20 gives richer interactions
p = 0.25
steps = 1200           # was 300; longer reduces noise
burn_in = 200          # ignore early transient
runs_per_alpha = 30    # was 8; more runs = smoother curve
seed_base = 2026

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

def gap_ahead(pos, lane_cells):
    for d in range(1, L):
        j = (pos + d) % L
        if lane_cells[j] is not None:
            return d - 1
    return L - 1

def jam_clusters(cars):
    jam = [0] * L
    for c in cars:
        if c.speed <= 1:
            jam[c.pos] = 1
    if sum(jam) == 0:
        return 0
    clusters = 0
    for i in range(L):
        if jam[i] == 1 and jam[i - 1] == 0:
            clusters += 1
    return clusters

def run_once(alpha, seed):
    random.seed(seed)

    lane_cells = [None] * L

    # evenly spaced around the ring
    spacing = L // num_cars
    positions = [(i * spacing) % L for i in range(num_cars)]

    k_auto = int(round(alpha * num_cars))
    cars = []
    for i, pos in enumerate(positions):
        is_human = (i >= k_auto)
        c = Cars(v_max=v_max, current_edge=(0, 0), human=is_human, lane=0, pos=pos)
        cars.append(c)
        lane_cells[pos] = c

    slow_counts = []
    cluster_counts = []

    for t in range(steps):
        if t >= burn_in:
            slow_counts.append(sum(1 for c in cars if c.speed <= 1))
            cluster_counts.append(jam_clusters(cars))

        # remove all cars
        for c in cars:
            lane_cells[c.pos] = None

        # update back-to-front
        for c in sorted(cars, key=lambda x: x.pos, reverse=True):
            c.accelarate()
            gap = gap_ahead(c.pos, lane_cells)
            c.brake(gap)

            # human randomness only
            if c.isHuman:
                c.randomise(p)

            c.move()
            c.pos %= L
            lane_cells[c.pos] = c

    return float(np.mean(slow_counts)), float(np.mean(cluster_counts))

# ----------------------------
# Run sweep + collect stats
# ----------------------------
results = []
print("alpha | mean_slow | sd_slow | mean_clusters | sd_clusters")
print("--------------------------------------------------------")

for alpha in alphas:
    slow_list = []
    clus_list = []

    for r in range(runs_per_alpha):
        seed = seed_base + r + int(alpha * 1000)
        avg_slow, avg_clus = run_once(alpha, seed)
        slow_list.append(avg_slow)
        clus_list.append(avg_clus)

    mean_slow = float(np.mean(slow_list))
    sd_slow = float(np.std(slow_list, ddof=1))
    mean_clus = float(np.mean(clus_list))
    sd_clus = float(np.std(clus_list, ddof=1))

    results.append((alpha, mean_slow, sd_slow, mean_clus, sd_clus))
    print(f"{alpha:>4.2f} | {mean_slow:>9.3f} | {sd_slow:>7.3f} | {mean_clus:>12.3f} | {sd_clus:>10.3f}")

# ----------------------------
# Save to CSV
# ----------------------------
csv_path = "alpha_sweep_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["alpha", "mean_slow_cars", "sd_slow_cars", "mean_jam_clusters", "sd_jam_clusters"])
    writer.writerows(results)

print(f"\nSaved results to: {csv_path}")

# ----------------------------
# Plot with error bars (standard error)
# ----------------------------
alpha_vals = np.array([r[0] for r in results])
mean_slow = np.array([r[1] for r in results])
sd_slow = np.array([r[2] for r in results])
mean_clus = np.array([r[3] for r in results])
sd_clus = np.array([r[4] for r in results])

se_slow = sd_slow / np.sqrt(runs_per_alpha)
se_clus = sd_clus / np.sqrt(runs_per_alpha)

plt.figure(figsize=(8, 5))
plt.errorbar(alpha_vals, mean_clus, yerr=se_clus, fmt="o-", capsize=4)
plt.xlabel("α (fraction of self-driving cars)")
plt.ylabel("Average jam clusters (speed ≤ 1)")
plt.title("Phantom jam frequency vs self-driving percentage")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.errorbar(alpha_vals, mean_slow, yerr=se_slow, fmt="o-", capsize=4)
plt.xlabel("α (fraction of self-driving cars)")
plt.ylabel("Average slow cars (speed ≤ 1)")
plt.title("Traffic instability vs self-driving percentage")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
