import random
import scipy.io
import networkx as nx

# ----------------------------
# Settings
# ----------------------------
L = 30          # <-- shorter roads to force interactions
subgraph_nodes = [0, 4, 14, 21, 22, 24, 25, 29]
N_CARS = 120    # <-- higher density
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
# Build roads: 1 lane per edge for now
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
cars = {}  # car_id -> dict(edge=(u,v), lane=int, i=int, is_auto=bool)

def try_spawn_car(car_id, max_tries=5000) -> bool:
    for _ in range(max_tries):
        edge = random.choice(edges)
        lane = 0
        i = random.randrange(0, L)
        if lane_is_empty(edge, lane, i):
            cars[car_id] = {
                "edge": edge,
                "lane": lane,
                "i": i,
                "is_auto": (random.random() < 0.4),
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
print("Total cells available:", len(edges) * 1 * L)

# ----------------------------
# Routing
# ----------------------------
def choose_next_edge(node_v):
    outs = list(G_sub.out_edges(node_v))
    if not outs:
        return None
    return random.choice(outs)

# ----------------------------
# Metrics: jam clusters
# A jam cell = occupied cell whose next cell is also occupied (no gap).
# A jam cluster = consecutive run of jam cells.
# ----------------------------
def jam_metrics():
    jammed_cars = 0
    jam_clusters = 0

    for edge, lanes in roads.items():
        for lane in lanes:
            # lane is a list of car_ids or None
            jam_flags = [False] * L
            for i in range(L - 1):
                if lane[i] is not None and lane[i + 1] is not None:
                    jam_flags[i] = True

            # count clusters in jam_flags
            in_cluster = False
            for i in range(L):
                if jam_flags[i]:
                    jammed_cars += 1
                    if not in_cluster:
                        jam_clusters += 1
                        in_cluster = True
                else:
                    in_cluster = False

    return jammed_cars, jam_clusters

# ----------------------------
# One synchronous step
# ----------------------------
def step():
    moves = []

    for car_id, st in cars.items():
        (u, v) = st["edge"]
        lane = st["lane"]
        i = st["i"]

        if i < L - 1:
            if lane_is_empty((u, v), lane, i + 1):
                moves.append((car_id, (u, v), lane, i, (u, v), lane, i + 1))
        else:
            nxt = choose_next_edge(v)
            if nxt is None:
                continue
            if lane_is_empty(nxt, 0, 0):
                moves.append((car_id, (u, v), lane, i, nxt, 0, 0))

    moves.sort(key=lambda x: x[0])
    occupied_targets = set()

    executed = 0
    for (car_id, fe, fl, fi, te, tl, ti) in moves:
        tgt = (te, tl, ti)
        if tgt in occupied_targets:
            continue
        if roads[te][tl][ti] is not None:
            continue

        remove_car(fe, fl, fi)
        place_car(te, tl, ti, car_id)
        cars[car_id]["edge"] = te
        cars[car_id]["lane"] = tl
        cars[car_id]["i"] = ti

        occupied_targets.add(tgt)
        executed += 1

    return executed

# ----------------------------
# Run sim
# ----------------------------
T = 200
for t in range(T):
    moved = step()
    if t % 20 == 0:
        jammed, clusters = jam_metrics()
        print(f"t={t:3d} moved={moved:3d} jammed_cars={jammed:3d} jam_clusters={clusters:3d}")

print("Done.")
