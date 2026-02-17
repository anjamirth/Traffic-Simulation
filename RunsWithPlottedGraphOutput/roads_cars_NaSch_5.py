import random
import scipy.io
import networkx as nx

# ----------------------------
# Settings
# ----------------------------
L = 30
subgraph_nodes = [0, 4, 14, 21, 22, 24, 25, 29]
SEED = 1

N_CARS = 160
v_max = 5

p = 0.30
alpha = 0.8
auto_frac = 0.4

T = 300
PRINT_EVERY = 20

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
    raise RuntimeError("Subgraph has no edges.")

roads = {(u, v): [[None] * L] for (u, v) in edges}

def lane_cells(edge):
    return roads[edge][0]

def is_empty(edge, pos):
    return lane_cells(edge)[pos] is None

def place(edge, pos, car_id):
    lane_cells(edge)[pos] = car_id

def remove(edge, pos):
    lane_cells(edge)[pos] = None

def choose_next_edge(node_v):
    outs = list(G_sub.out_edges(node_v))
    if not outs:
        return None
    return random.choice(outs)

# ----------------------------
# Cars
# ----------------------------
cars = {}

def try_spawn_car(car_id, max_tries=20000):
    for _ in range(max_tries):
        edge = random.choice(edges)
        pos = random.randrange(0, L)
        if is_empty(edge, pos):
            cars[car_id] = {
                "edge": edge,
                "pos": pos,
                "speed": random.randrange(0, v_max + 1),
                "is_auto": (random.random() < auto_frac),
            }
            place(edge, pos, car_id)
            return True
    return False

spawned = 0
for cid in range(N_CARS):
    if try_spawn_car(cid):
        spawned += 1

total_cells = len(edges) * L
print(f"Spawned cars: {spawned}/{N_CARS}   (density={spawned/total_cells:.3f} per cell)")
print("Edges:", len(edges), "Total cells:", total_cells)

# ----------------------------
# Jam metrics (occupancy adjacency)
# ----------------------------
def jam_metrics():
    jammed = 0
    clusters = 0
    for e in edges:
        lane = lane_cells(e)
        jam_flags = [False] * L
        for i in range(L - 1):
            if lane[i] is not None and lane[i + 1] is not None:
                jam_flags[i] = True

        in_cluster = False
        for i in range(L):
            if jam_flags[i]:
                jammed += 1
                if not in_cluster:
                    clusters += 1
                    in_cluster = True
            else:
                in_cluster = False
    return jammed, clusters

# ----------------------------
# NaSch update per edge, with FIXED boundary exiting
# Each exit event: (car_id, from_edge, node_v, next_edge_or_None)
# ----------------------------
def update_edge_nasch(edge):
    (u, v) = edge
    lane = lane_cells(edge)

    car_ids = [cid for cid in lane if cid is not None]
    if not car_ids:
        return [], 0

    car_ids.sort(key=lambda cid: cars[cid]["pos"], reverse=True)

    # clear lane
    for cid in car_ids:
        remove(edge, cars[cid]["pos"])

    exits = []
    moved_count = 0

    for idx, cid in enumerate(car_ids):
        st = cars[cid]
        pos = st["pos"]
        speed = st["speed"]
        is_auto = st["is_auto"]

        # Pick intended next edge only if we are at the boundary cell
        intended_next = None
        can_exit = False
        if pos == L - 1:
            intended_next = choose_next_edge(v)
            if intended_next is not None and is_empty(intended_next, 0):
                can_exit = True

        # gap to next car ahead (within same edge)
        if idx == 0:
            gap = (L - 1) - pos
        else:
            front_cid = car_ids[idx - 1]
            front_pos = cars[front_cid]["pos"]
            gap = (front_pos - pos) - 1

        # --- FIX: if at last cell and can_exit, give effective gap 1
        if pos == L - 1 and can_exit:
            gap = 1

        # 1) accelerate
        speed = min(speed + 1, v_max)

        # 2) brake to gap
        speed = min(speed, gap)

        # 3) randomise
        brake_p = p * (1 - alpha) if is_auto else p
        if speed > 0 and random.random() < brake_p:
            speed -= 1

        # 4) move
        new_pos = pos + speed
        if speed > 0:
            moved_count += 1

        if new_pos <= L - 1:
            st["pos"] = new_pos
            st["speed"] = speed
            place(edge, new_pos, cid)
        else:
            # exit (should only happen from last cell with gap=1)
            exits.append((cid, edge, v, intended_next))
            # Do NOT place it yet; process_exits will.

    return exits, moved_count

# ----------------------------
# Process exits: place car at cell0 of intended_next if still free
# Otherwise put it back at end of from_edge
# ----------------------------
def process_exits(exits):
    entered = 0
    queued = 0

    exits.sort(key=lambda x: x[0])

    for cid, from_edge, node_v, intended_next in exits:
        st = cars[cid]

        # intended_next might now be blocked; re-check
        if intended_next is not None and is_empty(intended_next, 0):
            st["edge"] = intended_next
            st["pos"] = 0
            st["speed"] = 0
            place(intended_next, 0, cid)
            entered += 1
        else:
            # queue back at end of from_edge
            st["edge"] = from_edge
            st["pos"] = L - 1
            st["speed"] = 0

            if is_empty(from_edge, L - 1):
                place(from_edge, L - 1, cid)
            else:
                # nearest empty backwards
                lane = lane_cells(from_edge)
                for j in range(L - 2, -1, -1):
                    if lane[j] is None:
                        st["pos"] = j
                        place(from_edge, j, cid)
                        break
            queued += 1

    return entered, queued

def step():
    all_exits = []
    moved_total = 0
    for e in sorted(edges):
        exits, moved = update_edge_nasch(e)
        all_exits.extend(exits)
        moved_total += moved
    entered, queued = process_exits(all_exits)
    return moved_total, entered, queued

# ----------------------------
# Run
# ----------------------------
for t in range(T):
    moved, entered, queued = step()
    if t % PRINT_EVERY == 0:
        jammed, clusters = jam_metrics()
        print(f"t={t:3d} moved={moved:3d} entered={entered:3d} queued={queued:3d} "
              f"jammed_cells={jammed:3d} jam_clusters={clusters:3d}")

print("Done.")
