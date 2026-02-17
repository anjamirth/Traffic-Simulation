print("=== Step 2E starting ===")

from Cars import Cars
import scipy.io
import networkx as nx
import random

L = 100
subgraph_nodes = [25, 0, 4, 14, 21, 22, 24, 29]
test_edge = (25, 22)
lane = 0

# Params
p = 0.25            # human randomness
steps = 80
num_cars = 15
spacing = 5         # initial spacing between cars
seed = 1
random.seed(seed)

data = scipy.io.loadmat("traffic_dataset.mat")
adj = data["tra_adj_mat"]
G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
G_sub = G.subgraph(subgraph_nodes).copy()

roads = {(u, v): [[None] * L] for (u, v) in G_sub.edges()}
lane_cells = roads[test_edge][lane]

# --- Spawn cars ---
cars = []
start_positions = [i * spacing for i in range(num_cars)]
start_positions = [pos for pos in start_positions if pos < L]

for pos in start_positions:
    c = Cars(v_max=5, current_edge=test_edge, human=True, lane=lane, pos=pos)
    cars.append(c)
    lane_cells[pos] = c

def find_gap(pos, lane_cells):
    for j in range(pos + 1, L):
        if lane_cells[j] is not None:
            return j - pos - 1
    return (L - 1) - pos

def jam_clusters(cars_sorted):
    """
    A jam-car = speed <= 1.
    Cluster = consecutive jam-cars with no gap > 1 cell between them.
    """
    jam_positions = [c.pos for c in cars_sorted if c.speed <= 1]
    if not jam_positions:
        return 0
    jam_positions.sort()
    clusters = 1
    for i in range(1, len(jam_positions)):
        if jam_positions[i] - jam_positions[i-1] > 1:
            clusters += 1
    return clusters

def print_lane(t):
    view_len = 80
    s = "".join("C" if x is not None else "." for x in lane_cells[:view_len])
    print(f"t={t:02d} {s}")

for t in range(steps + 1):
    # Print lane occasionally
    if t % 10 == 0:
        print_lane(t)

    # --- Measure jams ---
    cars_sorted = sorted(cars, key=lambda c: c.pos)
    num_slow = sum(1 for c in cars if c.speed <= 1)
    clusters = jam_clusters(cars_sorted)

    if t % 10 == 0:
        print(f"   slow_cars(speed<=1)={num_slow:02d}  jam_clusters={clusters}")

    # --- Update all cars from front to back ---
    cars_sorted = sorted(cars, key=lambda c: c.pos, reverse=True)

    # remove all cars from grid
    for c in cars:
        lane_cells[c.pos] = None

    # update each car
    for c in cars_sorted:
        c.accelarate()
        gap = find_gap(c.pos, lane_cells)
        c.brake(gap)
        c.randomise(p)
        c.move()

        # boundary clamp (temporary; we’ll replace with wraparound next)
        if c.pos >= L:
            c.pos = L - 1
            c.speed = 0

        lane_cells[c.pos] = c

print("=== Step 2E done ===")
