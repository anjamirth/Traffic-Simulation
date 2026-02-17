import random
import scipy.io
import networkx as nx
import statistics as stats

# ----------------------------
# Settings (keep fixed for fair comparison)
# ----------------------------
L = 30
subgraph_nodes = [0, 4, 14, 21, 22, 24, 25, 29]
SEED = 1

N_CARS = 160
v_max = 5

p = 0.30
auto_frac = 0.4

T = 500
WARMUP = 50  # discard first 50 steps from stats

ALPHAS = [0.0, 0.8]

# ----------------------------
# Load dataset + build subgraph once
# ----------------------------
data = scipy.io.loadmat("traffic_dataset.mat")
adj = data["tra_adj_mat"]
G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
G_sub = G.subgraph(subgraph_nodes).copy()
edges = list(G_sub.edges())
if not edges:
    raise RuntimeError("Subgraph has no edges.")


def run_sim(alpha: float, seed: int):
    random.seed(seed)

    # roads[(u,v)][0][pos] holds car_id or None
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

    # cars: id -> edge,pos,speed,is_auto
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

    # ----------------------------
    # Metrics: jam clusters (occupancy adjacency)
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
    # NaSch update per edge (single lane) with fixed boundary exiting
    # exit event: (car_id, from_edge, end_node_v, intended_next_edge)
    # ----------------------------
    def update_edge_nasch(edge):
        (u, v) = edge
        lane = lane_cells(edge)

        car_ids = [cid for cid in lane if cid is not None]
        if not car_ids:
            return [], 0

        car_ids.sort(key=lambda cid: cars[cid]["pos"], reverse=True)

        # clear lane; re-place after update / exit processing
        for cid in car_ids:
            remove(edge, cars[cid]["pos"])

        exits = []
        moved_count = 0

        for idx, cid in enumerate(car_ids):
            st = cars[cid]
            pos = st["pos"]
            speed = st["speed"]
            is_auto = st["is_auto"]

            intended_next = None
            can_exit = False
            if pos == L - 1:
                intended_next = choose_next_edge(v)
                if intended_next is not None and is_empty(intended_next, 0):
                    can_exit = True

            # gap to next car ahead (within edge)
            if idx == 0:
                gap = (L - 1) - pos
            else:
                front_cid = car_ids[idx - 1]
                front_pos = cars[front_cid]["pos"]
                gap = (front_pos - pos) - 1

            # FIX: if at boundary and can exit, allow one cell "virtual gap"
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
                exits.append((cid, edge, v, intended_next))

        return exits, moved_count

    def process_exits(exits):
        entered = 0
        queued = 0
        exits.sort(key=lambda x: x[0])

        for cid, from_edge, node_v, intended_next in exits:
            st = cars[cid]

            if intended_next is not None and is_empty(intended_next, 0):
                st["edge"] = intended_next
                st["pos"] = 0
                st["speed"] = 0
                place(intended_next, 0, cid)
                entered += 1
            else:
                # queue back on from_edge near end
                st["edge"] = from_edge
                st["pos"] = L - 1
                st["speed"] = 0

                if is_empty(from_edge, L - 1):
                    place(from_edge, L - 1, cid)
                else:
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
        jammed_cells, jam_clusters = jam_metrics()
        return moved_total, entered, queued, jammed_cells, jam_clusters

    # ----------------------------
    # Run and log every timestep
    # ----------------------------
    series = {
        "moved": [],
        "entered": [],
        "queued": [],
        "jammed_cells": [],
        "jam_clusters": [],
    }

    for _ in range(T):
        moved, entered, queued, jammed, clusters = step()
        series["moved"].append(moved)
        series["entered"].append(entered)
        series["queued"].append(queued)
        series["jammed_cells"].append(jammed)
        series["jam_clusters"].append(clusters)

    meta = {
        "alpha": alpha,
        "seed": seed,
        "spawned": spawned,
        "density": spawned / (len(edges) * L),
        "edges": len(edges),
        "cells": len(edges) * L,
    }
    return meta, series


def summarize(series, warmup):
    def sstats(x):
        x2 = x[warmup:]
        return {
            "mean": stats.mean(x2),
            "stdev": stats.pstdev(x2),  # population stdev (stable for reporting)
            "min": min(x2),
            "max": max(x2),
        }

    return {k: sstats(v) for k, v in series.items()}


# ----------------------------
# Run comparison
# ----------------------------
print("=== NaSch graph model: logging + summary stats ===")
print(f"Nodes: {len(subgraph_nodes)}  Edges: {len(edges)}  L={L}  T={T}  Warmup={WARMUP}")
print(f"N_CARS={N_CARS}  v_max={v_max}  p={p}  auto_frac={auto_frac}  seed={SEED}")
print()

results = []
for a in ALPHAS:
    meta, series = run_sim(alpha=a, seed=SEED)
    summ = summarize(series, WARMUP)
    results.append((meta, summ))

    print(f"--- alpha={a:.2f} ---")
    print(f"spawned={meta['spawned']} density={meta['density']:.3f} (cells={meta['cells']})")
    for key in ["moved", "entered", "queued", "jammed_cells", "jam_clusters"]:
        st = summ[key]
        print(f"{key:12s} mean={st['mean']:.2f}  sd={st['stdev']:.2f}  min={st['min']}  max={st['max']}")
    print()

#
#
#
#Autonomy reduces instability but cannot eliminate congestion caused by network topology.
#
#
#