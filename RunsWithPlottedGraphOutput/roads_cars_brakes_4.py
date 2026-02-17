import random
import scipy.io
import networkx as nx

# ----------------------------
# Settings
# ----------------------------
L = 30
subgraph_nodes = [0, 4, 14, 21, 22, 24, 25, 29]
N_CARS = 160
SEED = 1

p = 0.30       # human random brake probability
alpha = 0.0    # autonomy strength (0=none, 1=perfect)
auto_frac = 0.4

random.seed(SEED)

# ----------------------------
# Load dataset + build graph
# ----------------------------
data = scipy.io.loadmat("traffic_dataset.mat")
adj = data["tra_adj_mat"]
G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
G_sub = G.subgraph(subgraph_nodes).copy()

edges = list(G_sub.edges())
roads = {(u, v): [[None] * L] for (u, v) in edges}

def lane_is_empty(edge, lane_idx, cell_idx) -> bool:
    return roads[edge][lane_idx][cell_idx] is None

def place_car(edge, lane_idx, cell_idx, car_id):
    roads[edge][lane_idx][cell_idx] = car_id

def remove_car(edge, lane_idx, cell_idx):
    roads[edge][lane_idx][cell_idx] = None

# ----------------------------
# Cars
# ----------------------------
cars = {}  # id -> edge, lane, i, is_auto

def try_spawn_car(car_id, max_tries=10000) -> bool:
    for _ in range(max_tries):
        edge = random.choice(edges)
        i = random.randrange(0, L)
        if lane_is_empty(edge, 0, i):
            is_auto = (random.random() < auto_frac)
            cars[car_id] = {"edge": edge, "lane": 0, "i": i, "is_auto": is_auto}
            place_car(edge, 0, i, car_id)
            return True
    return False

spawned = 0
for cid in range(N_CARS):
    if try_spawn_car(cid):
        spawned += 1

print(f"Spawned cars: {spawned}/{N_CARS}   (density={spawned/(len(edges)*L):.3f} per cell)")
print("Edges:", len(edges), "Total cells:", len(edges) * L)

# ----------------------------
# Routing
# ----------------------------
def choose_next_edge(node_v):
    outs = list(G_sub.out_edges(node_v))
    if not outs:
        return None
    return random.choice(outs)

# ----------------------------
# Jam metrics
# ----------------------------
def jam_metrics():
    jammed_cars = 0
    jam_clusters = 0

    for lanes in roads.values():
        for lane in lanes:
            jam_flags = [False] * L
            for i in range(L - 1):
                if lane[i] is not None and lane[i + 1] is not None:
                    jam_flags[i] = True

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
# Step with random braking
# ----------------------------
def step():
    moves = []

    for car_id, st in cars.items():
        (u, v) = st["edge"]
        lane = st["lane"]
        i = st["i"]
        is_auto = st["is_auto"]

        # compute brake probability
        brake_p = p * (1 - alpha) if is_auto else p

        # Try to move forward by 1 if possible
        if i < L - 1:
            if lane_is_empty((u, v), lane, i + 1):
                # random braking: sometimes choose not to move
                if random.random() >= brake_p:
                    moves.append((car_id, (u, v), lane, i, (u, v), lane, i + 1))
        else:
            # edge end: attempt to enter next edge cell 0
            nxt = choose_next_edge(v)
            if nxt is None:
                continue
            if lane_is_empty(nxt, 0, 0):
                # braking at junction too (humans hesitate)
                if random.random() >= brake_p:
                    moves.append((car_id, (u, v), lane, i, nxt, 0, 0))

    # conflict resolution
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
# Run
# ----------------------------
T = 300
for t in range(T):
    moved = step()
    if t % 20 == 0:
        jammed, clusters = jam_metrics()
        print(f"t={t:3d} moved={moved:3d} jammed_cars={jammed:3d} jam_clusters={clusters:3d}")

print("Done.")
