from Cars import Cars
import random
import numpy as np

# ----------------------------
# Ring-road parameters
# ----------------------------
L = 100
v_max = 5
num_cars = 15
p = 0.25
steps = 300
burn_in = 50       # ignore first 50 steps (startup transient)
runs_per_alpha = 8
seed_base = 100

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

    # lane occupancy
    lane_cells = [None] * L

    # evenly spaced positions
    spacing = L // num_cars
    positions = [(i * spacing) % L for i in range(num_cars)]

    # create cars: first k are autonomous, rest human
    k_auto = int(round(alpha * num_cars))
    cars = []
    for i, pos in enumerate(positions):
        is_human = (i >= k_auto)
        c = Cars(v_max=v_max, current_edge=(0,0), human=is_human, lane=0, pos=pos)
        cars.append(c)
        lane_cells[pos] = c

    slow_counts = []
    cluster_counts = []

    for t in range(steps):
        # measure after burn-in
        if t >= burn_in:
            slow_counts.append(sum(1 for c in cars if c.speed <= 1))
            cluster_counts.append(jam_clusters(cars))

        # remove all cars
        for c in cars:
            lane_cells[c.pos] = None

        # update cars (back-to-front)
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

print("alpha | avg_slow_cars | avg_jam_clusters")
print("----------------------------------------")

for alpha in alphas:
    slow_list = []
    clus_list = []
    for r in range(runs_per_alpha):
        avg_slow, avg_clus = run_once(alpha, seed_base + r + int(alpha*1000))
        slow_list.append(avg_slow)
        clus_list.append(avg_clus)

    print(f"{alpha:>4.2f} | {np.mean(slow_list):>13.3f} | {np.mean(clus_list):>16.3f}")
