print("=== Step 2C starting ===")

from Cars import Cars
import scipy.io
import networkx as nx

L = 100
subgraph_nodes = [25, 0, 4, 14, 21, 22, 24, 29]

# Parameters
p = 0.2          # human random slowdown probability
steps = 10       # number of timesteps to simulate

data = scipy.io.loadmat("traffic_dataset.mat")
adj = data["tra_adj_mat"]

G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
G_sub = G.subgraph(subgraph_nodes).copy()

roads = {(u, v): [[None] * L] for (u, v) in G_sub.edges()}

test_edge = (25, 22)
lane = 0

# Spawn 1 human car at position 0
car = Cars(v_max=5, current_edge=test_edge, human=True, lane=lane, pos=0)
roads[test_edge][lane][car.pos] = car

def print_lane(t):
    view_len = 40
    lane_view = roads[test_edge][lane][:view_len]
    s = "".join("C" if cell is not None else "." for cell in lane_view)
    print(f"t={t:02d} pos={car.pos:02d} speed={car.speed}  {s}")

print_lane(0)

for t in range(1, steps + 1):
    # Remove from old cell
    roads[test_edge][lane][car.pos] = None

    # Update speed (no braking because no other cars yet)
    car.accelarate()
    car.randomise(p)
    car.move()

    # Keep within edge bounds (simple clamp for now)
    if car.pos >= L:
        car.pos = L - 1
        car.speed = 0

    # Place back into grid
    roads[test_edge][lane][car.pos] = car

    print_lane(t)

print("=== Step 2C done ===")
