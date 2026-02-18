import os
import math
import statistics as stats
import matplotlib.pyplot as plt

from sim_model import build_subgraph, run_sim, summarize

# ----------------------------
# Settings
# ----------------------------
mat_path = "traffic_dataset.mat"
subgraph_nodes = [0, 4, 14, 21, 22, 24, 25, 29]

road_length = 30
num_vehicles = 160
max_speed = 5

human_brake_prob = 0.30
random_braking_reduction = 0.80

num_timesteps = 500
warmup_steps = 50

penetration_rates = [0.0, 0.25, 0.50, 0.75, 1.0]
seeds = [1, 2, 3, 4, 5]

# ----------------------------
# Build graph once
# ----------------------------
G_sub, edges = build_subgraph(mat_path, subgraph_nodes)

print("=== Multi-seed Sweep penetration_rate (random_braking_reduction fixed) ===")
print(f"Nodes={len(subgraph_nodes)} Edges={len(edges)} road_length={road_length} num_timesteps={num_timesteps} warmup_steps={warmup_steps}")
print(f"num_vehicles={num_vehicles} max_speed={max_speed} human_brake_prob={human_brake_prob} random_braking_reduction={random_braking_reduction}")
print(f"Seeds={seeds}\n")

results = {
    "penetration_rate": [],
    "moved_mean": [],
    "moved_sd": [],
    "jam_mean": [],
    "jam_sd": [],
    "CV_mean": [],
    "nmse_mean": []
}

for pr in penetration_rates:
    agg = {"moved": [], "jam_clusters": [], "CV": [], "normalized_MSE": []}

    for seed in seeds:
        meta, series = run_sim(
            G_sub=G_sub,
            edges=edges,
            road_length=road_length,
            num_vehicles=num_vehicles,
            max_speed=max_speed,
            human_brake_prob=human_brake_prob,
            random_braking_reduction=random_braking_reduction,
            penetration_rate=pr,
            num_timesteps=num_timesteps,
            seed=seed,
        )
        summ = summarize(series, warmup_steps)

        agg["moved"].append(summ["moved"]["mean"])
        agg["jam_clusters"].append(summ["jam_clusters"]["mean"])
        agg["CV"].append(summ["predictability"]["CV"])
        agg["normalized_MSE"].append(summ["predictability"]["normalized_MSE"])

    mean_moved = stats.mean(agg["moved"])
    sd_moved = stats.pstdev(agg["moved"])
    mean_jam = stats.mean(agg["jam_clusters"])
    sd_jam = stats.pstdev(agg["jam_clusters"])
    mean_cv = stats.mean(agg["CV"])
    mean_nmse = stats.mean(agg["normalized_MSE"])

    results["penetration_rate"].append(pr)

    results["moved_mean"].append(mean_moved)
    results["moved_sd"].append(sd_moved)

    results["jam_mean"].append(mean_jam)
    results["jam_sd"].append(sd_jam)

    results["CV_mean"].append(mean_cv)
    results["nmse_mean"].append(mean_nmse)


    print(f"--- penetration_rate={pr:.2f} ---")
    print(f"moved mean={mean_moved:.2f}  sd_across_seeds={sd_moved:.2f}")
    print(f"jam_clusters mean={mean_jam:.2f}  sd_across_seeds={sd_jam:.2f}")
    print(f"CV mean={mean_cv:.4f}")
    print(f"normalized_MSE mean={mean_nmse:.6f}\n")

def plot_with_ci(ax, x, means, stds, n_seeds, title, ylabel):
    """
    Plot mean line with 95% confidence interval band.
    """
    ax.plot(x, means, marker="o")

    # Standard error
    se = [sd / math.sqrt(n_seeds) for sd in stds]

    # 95% CI
    lower = [m - 1.96 * s for m, s in zip(means, se)]
    upper = [m + 1.96 * s for m, s in zip(means, se)]

    ax.fill_between(x, lower, upper, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel("Autonomous Penetration Rate")
    ax.set_ylabel(ylabel)
    ax.grid(True)


# ----------------------------
#               Plot
# ----------------------------
x = results["penetration_rate"]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.canvas.manager.set_window_title("Autonomous Network Summary")
fig.suptitle("Impact of Autonomous Vehicle Penetration on Network Dynamics", fontsize=14)

plot_with_ci(
    axs[0, 0],
    x,
    results["jam_mean"],
    results["jam_sd"],
    len(seeds),
    "Jam Clusters",
    "Mean"
)

plot_with_ci(
    axs[0, 1],
    x,
    results["moved_mean"],
    results["moved_sd"],
    len(seeds),
    "Throughput (Moved)",
    "Mean Cars Moved / Timestep"
)

axs[1, 0].plot(x, results["CV_mean"], marker="o")
axs[1, 0].set_title("Flow CV (entered)")
axs[1, 0].set_xlabel("Autonomous Penetration Rate")
axs[1, 0].set_ylabel("σ / μ")

axs[1, 1].plot(x, results["nmse_mean"], marker="o")
axs[1, 1].set_title("Normalized Forecast Error")
axs[1, 1].set_xlabel("Autonomous Penetration Rate")
axs[1, 1].set_ylabel("MSE / μ²")

for ax in axs.flat:
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])

os.makedirs("output", exist_ok=True)
plt.savefig("output/auto_penetration_summary.png", dpi=300, bbox_inches="tight")
plt.show()
