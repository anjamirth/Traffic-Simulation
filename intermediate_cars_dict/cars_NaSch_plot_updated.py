import os
import random
import scipy.io
import networkx as nx
import statistics as stats
import matplotlib.pyplot as plt

from Cars import Cars

# --------------------------------------------------------
#   Fixed model settings (keep constant across sweep)
# --------------------------------------------------------
road_length = 30
subgraph_nodes = [0, 4, 14, 21, 22, 24, 25, 29]

random_seeds = [1, 2, 3, 4, 5]

no_of_cars = 160
v_max = 5

human_brake_prob = 0.30
random_braking_reduction = 0.80   # how much autonomous cars reduce random braking (0..1)

num_timesteps = 500
warmup_steps = 50

penetration_rates = [0.0, 0.25, 0.50, 0.75, 1.0]  # % autonomous cars

# --------------------------------------------------------
#           Load dataset + build subgraph once
# --------------------------------------------------------
data = scipy.io.loadmat("traffic_dataset.mat")
adj = data["tra_adj_mat"]
G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
G_sub = G.subgraph(subgraph_nodes).copy()

edges = list(G_sub.edges())
if not edges:
    raise RuntimeError("Subgraph has no edges.")

def run_sim(penetration_rate: float, seed: int):
    random.seed(seed)

    # Each edge has 1 lane, lane has road_length cells storing car_id or None
    roads = {(u, v): [[None] * road_length] for (u, v) in edges}

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

    # cars: car_id -> Cars object
    cars = {}

    def try_spawn_car(car_id, max_tries=20000):
        for _ in range(max_tries):
            edge = random.choice(edges)
            pos = random.randrange(0, road_length)
            if is_empty(edge, pos):
                speed0 = random.randrange(0, v_max + 1)
                is_auto = (random.random() < penetration_rate)
                is_human = (not is_auto)
                cars[car_id] = Cars(
                    v_max=v_max,
                    current_edge=edge,
                    is_human=is_human,
                    lane=0,
                    pos=pos,
                    speed=speed0
                )
                place(edge, pos, car_id)
                return True
        return False

    spawned = 0
    for cid in range(no_of_cars):
        if try_spawn_car(cid):
            spawned += 1

    def jam_metrics():
        jammed = 0
        clusters = 0
        for e in edges:
            lane = lane_cells(e)
            jam_flags = [False] * road_length

            # jam flag at i if two adjacent cells are occupied at i and i+1
            for i in range(road_length - 1):
                if lane[i] is not None and lane[i + 1] is not None:
                    jam_flags[i] = True

            in_cluster = False
            for i in range(road_length):
                if jam_flags[i]:
                    jammed += 1
                    if not in_cluster:
                        clusters += 1
                        in_cluster = True
                else:
                    in_cluster = False

        return jammed, clusters

    def update_edge_nasch(edge):
        (u, v) = edge
        lane = lane_cells(edge)

        car_ids = [cid for cid in lane if cid is not None]
        if not car_ids:
            return [], 0

        # sort from front to back (higher pos first)
        car_ids.sort(key=lambda cid: cars[cid].pos, reverse=True)

        # clear cars from this lane
        for cid in car_ids:
            remove(edge, cars[cid].pos)

        exits = []
        moved_count = 0

        for idx, cid in enumerate(car_ids):
            car = cars[cid]
            pos = car.pos

            intended_next = None
            can_exit = False

            # If at end of edge, try to reserve entry into next edge at cell 0
            if pos == road_length - 1:
                intended_next = choose_next_edge(v)
                if intended_next is not None and is_empty(intended_next, 0):
                    can_exit = True

            # gap to next car ahead (within this edge)
            if idx == 0:
                gap = (road_length - 1) - pos
            else:
                front_cid = car_ids[idx - 1]
                front_pos = cars[front_cid].pos
                gap = (front_pos - pos) - 1

            # if you can exit, allow 1-cell "gap" to step off the edge
            if pos == road_length - 1 and can_exit:
                gap = 1

            # NaSch update using Cars class
            car.accelerate()
            car.brake(gap)
            car.randomise(human_brake_prob, random_braking_reduction)

            new_pos = pos + car.speed
            if car.speed > 0:
                moved_count += 1

            if new_pos <= road_length - 1:
                car.pos = new_pos
                place(edge, new_pos, cid)
            else:
                exits.append((cid, edge, v, intended_next))

        return exits, moved_count

    def process_exits(exits):
        entered = 0
        queued = 0
        exits.sort(key=lambda x: x[0])

        for cid, from_edge, node_v, intended_next in exits:
            car = cars[cid]

            if intended_next is not None and is_empty(intended_next, 0):
                car.edge = intended_next
                car.pos = 0
                car.speed = 0
                place(intended_next, 0, cid)
                entered += 1
            else:
                # queue back onto from_edge near the end
                car.edge = from_edge
                car.speed = 0
                car.pos = road_length - 1

                if is_empty(from_edge, road_length - 1):
                    place(from_edge, road_length - 1, cid)
                else:
                    lane = lane_cells(from_edge)
                    for j in range(road_length - 2, -1, -1):
                        if lane[j] is None:
                            car.pos = j
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
        jammed, clusters = jam_metrics()
        return moved_total, entered, queued, jammed, clusters

    series = {k: [] for k in ["moved", "entered", "queued", "jammed_cells", "jam_clusters"]}

    for _ in range(num_timesteps):
        moved, entered, queued, jammed, clusters = step()
        series["moved"].append(moved)
        series["entered"].append(entered)
        series["queued"].append(queued)
        series["jammed_cells"].append(jammed)
        series["jam_clusters"].append(clusters)

    meta = {
        "penetration_rate": penetration_rate,
        "seed": seed,
        "spawned": spawned,
        "density": spawned / (len(edges) * road_length),
    }
    return meta, series


def summarize(series, warmup):
    def sstats(x):
        x2 = x[warmup:]
        return {
            "mean": stats.mean(x2),
            "sd": stats.pstdev(x2),
            "min": min(x2),
            "max": max(x2),
        }

    summary = {k: sstats(v) for k, v in series.items()}

    # Predictability metrics on entered(t)
    flow = series["entered"][warmup:]
    mean_flow = stats.mean(flow)
    sd_flow = stats.pstdev(flow)
    cv = sd_flow / mean_flow if mean_flow != 0 else 0

    # Lag-1 autocorrelation
    if len(flow) > 1:
        x = flow[:-1]
        y = flow[1:]
        mean_x = stats.mean(x)
        mean_y = stats.mean(y)
        cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / len(x)
        var_x = stats.pstdev(x) ** 2
        autocorr = cov / var_x if var_x != 0 else 0
    else:
        autocorr = 0

    # Naive forecast MSE and normalised MSE
    mse = sum((flow[i] - flow[i - 1]) ** 2 for i in range(1, len(flow))) / (len(flow) - 1)
    nmse = mse / (mean_flow ** 2) if mean_flow != 0 else 0

    summary["predictability"] = {
        "CV": cv,
        "lag1_autocorr": autocorr,
        "naive_MSE": mse,
        "normalized_MSE": nmse,
    }

    return summary


print("=== Multi-seed Sweep penetration_rate (random_braking_reduction fixed) ===")
print(f"Nodes={len(subgraph_nodes)} Edges={len(edges)} road_length={road_length} num_timesteps={num_timesteps} Warmup={warmup_steps}")
print(f"no_of_cars={no_of_cars} v_max={v_max} human_brake_prob={human_brake_prob} random_braking_reduction={random_braking_reduction}")
print(f"Seeds={random_seeds}")
print()

results = {
    "penetration_rate": [],
    "moved": [],
    "jam_clusters": [],
    "CV": [],
    "normalized_MSE": []
}

for pr in penetration_rates:
    agg = {"moved": [], "jam_clusters": [], "CV": [], "normalized_MSE": []}

    for seed in random_seeds:
        meta, series = run_sim(penetration_rate=pr, seed=seed)
        summ = summarize(series, warmup_steps)

        agg["moved"].append(summ["moved"]["mean"])
        agg["jam_clusters"].append(summ["jam_clusters"]["mean"])
        agg["CV"].append(summ["predictability"]["CV"])
        agg["normalized_MSE"].append(summ["predictability"]["normalized_MSE"])

    mean_moved = stats.mean(agg["moved"])
    sd_moved = stats.pstdev(agg["moved"])

    mean_jam = stats.mean(agg["jam_clusters"])
    sd_jam = stats.pstdev(agg["jam_clusters"])

    mean_cv = stats.mean(agg["CV"])
    mean_nmse = stats.mean(agg["normalized_MSE"])

    results["penetration_rate"].append(pr)
    results["moved"].append(mean_moved)
    results["jam_clusters"].append(mean_jam)
    results["CV"].append(mean_cv)
    results["normalized_MSE"].append(mean_nmse)

    print(f"--- penetration_rate={pr:.2f} ---")
    print(f"moved mean={mean_moved:.2f}  sd_across_seeds={sd_moved:.2f}")
    print(f"jam_clusters mean={mean_jam:.2f}  sd_across_seeds={sd_jam:.2f}")
    print(f"CV mean={mean_cv:.4f}")
    print(f"normalized_MSE mean={mean_nmse:.6f}")
    print()
    print()

# --------------------------------------------------------
#               Plot 2x2 summary figure
# --------------------------------------------------------
x = results["penetration_rate"]
moved = results["moved"]
jam_clusters = results["jam_clusters"]
CV = results["CV"]
normalized_MSE = results["normalized_MSE"]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.canvas.manager.set_window_title("Autonomous Network Summary")
fig.suptitle("Impact of Autonomous Vehicle Penetration on Network Dynamics", fontsize=14)

axs[0, 0].plot(x, jam_clusters, marker="o")
axs[0, 0].set_title("Jam Clusters")
axs[0, 0].set_xlabel("Autonomous Penetration Rate")
axs[0, 0].set_ylabel("Mean")

axs[0, 1].plot(x, moved, marker="o")
axs[0, 1].set_title("Throughput (Moved)")
axs[0, 1].set_xlabel("Autonomous Penetration Rate")
axs[0, 1].set_ylabel("Mean Cars Moved / Timestep")

axs[1, 0].plot(x, CV, marker="o")
axs[1, 0].set_title("Flow CV (entered)")
axs[1, 0].set_xlabel("Autonomous Penetration Rate")
axs[1, 0].set_ylabel("σ / μ")

axs[1, 1].plot(x, normalized_MSE, marker="o")
axs[1, 1].set_title("Normalized Forecast Error")
axs[1, 1].set_xlabel("Autonomous Penetration Rate")
axs[1, 1].set_ylabel("MSE / μ²")

for ax in axs.flat:
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])

os.makedirs("output", exist_ok=True)
plt.savefig("output/auto_penetration_summary.png", dpi=300, bbox_inches="tight")

plt.show()
