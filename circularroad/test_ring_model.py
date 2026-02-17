from ring_model import RingRoadModel
import numpy as np

# Parameters
L = 100
num_cars = 20
v_max = 5
p = 0.25
alpha = 0.0
seed = 10
steps = 500

model = RingRoadModel(
    L=L,
    num_cars=num_cars,
    v_max=v_max,
    p=p,
    alpha=alpha,
    seed=seed
)

slow_history = []
cluster_history = []

for _ in range(steps):
    stats = model.step(jam_threshold=1)
    slow_history.append(stats.slow_cars)
    cluster_history.append(stats.jam_clusters)

print("Finished simulation.\n")

print(f"Average slow cars: {np.mean(slow_history):.3f}")
print(f"Average jam clusters: {np.mean(cluster_history):.3f}")
print(f"Max jam clusters: {max(cluster_history)}")
print(f"Min jam clusters: {min(cluster_history)}")
