import random
import scipy.io
import networkx as nx

# ----------------------------
# Settings
# ----------------------------
L = 100  # cells per directed edge lane
subgraph_nodes = [0, 4, 14, 21, 22, 24, 25, 29]
N_CARS = 80
SEED = 1

random.seed(SEED)

# ----------------------------
# Load dataset + build graph
# ----------------------------
data = scipy.io.loadmat("traffic_dataset.mat")
adj = data["tra_adj_mat"]
G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
G_sub = G.subgraph(subgraph_nodes).copy()

edges = list(G_sub.edges())
if not edges:
    raise RuntimeError("Subgraph has no edges — pick different nodes.")

# ----------------------------
# Build roads: (u,v) -> list_of_lanes; each lane = [None]*L
# For now force 1 lane per edge (stable baseline).
# We can plug lane counts back in after we validate the correct feature column.
# ----------------------------
roads = {(u, v): [[None] * L] for (u, v) in edges}

def lane_is_empty(edge, lane_idx, cell_idx) -> bool:
    return roads[edge][lane_idx][cell_idx] is None

def place_car(edge, lane_idx, cell_idx, car_id):
    roads[edge][lane_idx][cell_idx] = car_id

def remove_car(edge, lane_idx, cell_idx):
    roads[edge][lane_idx][cell_idx] = None

# ----------------------------
# Car state
# ----------------------------
# Each car lives on exactly one edge lane at a cell index.
cars = {}  # car_id -> dict(edge=(u,v), lane=int, i=int, is_auto=bool)

def try_spawn_car(car_id, max_tries=2000) -> bool:
    for _ in range(max_tries):
        edge = random.choice(edges)
        lane = 0
        i = random.randrange(0, L)  # anywhere
        if lane_is_empty(edge, lane, i):
            cars[car_id] = {
                "edge": edge,
                "lane": lane,
                "i": i,
                "is_auto": (random.random() < 0.4),  # 40% autonomous placeholder
            }
            place_car(edge, lane, i, car_id)
            return True
    return False

spawned = 0
for car_id in range(N_CARS):
    if try_spawn_car(car_id):
        spawned += 1

print(f"Spawned cars: {spawned}/{N_CARS}")
print("Subgraph edges:", len(edges))

# ----------------------------
# Routing
# ----------------------------
def choose_next_edge(node_v):
    outs = list(G_sub.out_edges(node_v))
    if not outs:
        return None
    return random.choice(outs)  # (v, w)

# ----------------------------
# One synchronous step
# ----------------------------
def step():
    # Build proposed moves first (synchronous)
    moves = []  # (car_id, from_edge, lane, from_i, to_edge, to_lane, to_i)

    for car_id, st in cars.items():
        (u, v) = st["edge"]
        lane = st["lane"]
        i = st["i"]

        # If not at end of edge, try move forward by 1
        if i < L - 1:
            if lane_is_empty((u, v), lane, i + 1):
                moves.append((car_id, (u, v), lane, i, (u, v), lane, i + 1))
            # else: blocked, no move
        else:
            # At end of edge: try enter next edge at cell 0
            nxt = choose_next_edge(v)
            if nxt is None:
                # dead end: stay at end
                continue
            # For now: 1 lane only
            if lane_is_empty(nxt, 0, 0):
                moves.append((car_id, (u, v), lane, i, nxt, 0, 0))
            # else: queue at end

    # Resolve conflicts: if two cars target same cell, only one moves (deterministic)
    # Sort by car_id so result is stable.
    moves.sort(key=lambda x: x[0])
    occupied_targets = set()

    executed = 0
    for (car_id, fe, fl, fi, te, tl, ti) in moves:
        tgt = (te, tl, ti)
        if tgt in occupied_targets:
            continue
        # Also ensure still empty (in case another move filled it)
        if roads[te][tl][ti] is not None:
            continue

        # Execute move
        remove_car(fe, fl, fi)
        place_car(te, tl, ti, car_id)
        cars[car_id]["edge"] = te
        cars[car_id]["lane"] = tl
        cars[car_id]["i"] = ti

        occupied_targets.add(tgt)
        executed += 1

    return executed

# ----------------------------
# Run a short simulation and print simple congestion indicators
# ----------------------------
def count_blocked_at_ends():
    # Cars stuck at i==L-1
    return sum(1 for st in cars.values() if st["i"] == L - 1)

T = 200
for t in range(T):
    moved = step()
    blocked_ends = count_blocked_at_ends()
    if t % 20 == 0:
        print(f"t={t:3d} moved={moved:3d} blocked_at_edge_ends={blocked_ends:3d}")

print("Done.")
