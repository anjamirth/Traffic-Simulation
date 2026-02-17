import numpy as np
from ring_model import RingRoadModel

# ----------------------------
# Experiment settings
# ----------------------------
L = 100
num_cars = 20
v_max = 5
p = 0.25

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

steps = 1200
burn_in = 200          # ignore early settling
runs_per_alpha = 20    # increase for smoother stats
jam_threshold = 1

seed_start = 1         # seeds: 1..runs_per_alpha


def run_once(alpha: float, seed: int):
    model = RingRoadModel(L=L, num_cars=num_cars, v_max=v_max, p=p, alpha=alpha, seed=seed)

    slow = []
    clusters = []

    for t in range(steps):
        stats = model.step(jam_threshold=jam_threshold)
        if t >= burn_in:
            slow.append(stats.slow_cars)
            clusters.append(stats.jam_clusters)

    return float(np.mean(slow)), float(np.mean(clusters))


def main():
    print("alpha | mean_clusters | std_clusters | mean_slow_cars | std_slow_cars")
    print("---------------------------------------------------------------------")

    for alpha in alphas:
        slow_means = []
        cluster_means = []

        for r in range(runs_per_alpha):
            seed = seed_start + r
            mean_slow, mean_clusters = run_once(alpha, seed)
            slow_means.append(mean_slow)
            cluster_means.append(mean_clusters)

        # across-seed statistics
        mean_clusters = float(np.mean(cluster_means))
        std_clusters = float(np.std(cluster_means, ddof=1)) if runs_per_alpha > 1 else 0.0

        mean_slow = float(np.mean(slow_means))
        std_slow = float(np.std(slow_means, ddof=1)) if runs_per_alpha > 1 else 0.0

        print(f"{alpha:>4.2f} |"
              f" {mean_clusters:>12.3f} | {std_clusters:>11.3f} |"
              f" {mean_slow:>14.3f} | {std_slow:>12.3f}")


if __name__ == "__main__":
    main()
