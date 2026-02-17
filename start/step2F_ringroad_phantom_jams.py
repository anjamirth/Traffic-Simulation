print("=== Step 2F starting (RING ROAD) ===")

from Cars import Cars
import scipy.io
import networkx as nx
import random

L = 100
subgraph_nodes = [25, 0, 4, 14, 21, 22, 24, 29]
test_edge = (25, 22)
lane = 0

# Params
p = 0.25
steps = 120
num_cars = 15
seed = 1
random.seed(seed)

data = scipy.io.loadmat("traffic_dataset.mat")
adj = data["tra_adj_mat"]
G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
G_sub = G.subgraph(subgraph_nodes).copy()

roads = {(u, v): [[None] * L] for (u, v) in G_sub.edges()}
lane_cells = roads[test_edge][lane]

# --- Spawn cars evenly spaced around the ring ---
cars = []
spacing = L // num_cars  # even spacing
positions = [(i * spacing) % L for i in range(num_cars)]

for pos in positions:
    c = Cars(v_max=5, current_edge=test_edge, human=True, lane=lane, pos=pos)
    cars.append(c)
    lane_cells[pos] = c

def gap_ahead(pos, lane_cells):
    """Gap on a ring = distance to next car ahead (excluding next car cell)."""
    for d in range(1, L):  # look forward 1..L-1
        j = (pos + d) % L
        if lane_cells[j] is not None:
            return d - 1
    return L - 1  # should never happen unless only 1 car

def jam_clusters():
    """Count clusters of jam-cells (cars with speed<=1) around a ring."""
    jam = [0] * L
    for c in cars:
        if c.speed <= 1:
            jam[c.pos] = 1

    if sum(jam) == 0:
        return 0

    # Count transitions 0->1; ring means wrap-around matters
    clusters = 0
    for i in range(L):
        prev = jam[i - 1]  # works because Python wraps negative index
        if jam[i] == 1 and prev == 0:
            clusters += 1
    return clusters

def print_lane(t):
    view_len = 80
    s = "".join("C" if x is not None else "." for x in lane_cells[:view_len])
    num_slow = sum(1 for c in cars if c.speed <= 1)
    clusters = jam_clusters()
    print(f"t={t:03d} slow={num_slow:02d} clusters={clusters}  {s}")

for t in range(steps + 1):
    if t % 10 == 0:
        print_lane(t)

    # remove all cars
    for c in cars:
        lane_cells[c.pos] = None

    # update cars (order doesn't matter much on ring if we use occupancy checks)
    # but using forward-looking gaps requires cars to be placed as we go from front.
    # easiest: sort by position descending and place as we update.
    for c in sorted(cars, key=lambda x: x.pos, reverse=True):
        c.accelarate()
        gap = gap_ahead(c.pos, lane_cells)
        c.brake(gap)
        c.randomise(p)
        c.move()
        c.pos %= L  # wraparound
        lane_cells[c.pos] = c

print("=== Step 2F done ===")
