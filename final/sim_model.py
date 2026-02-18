import random
import scipy.io
import networkx as nx
import statistics as stats

from Cars import Cars

def build_subgraph(mat_path: str, subgraph_nodes):
    data = scipy.io.loadmat(mat_path)
    adj = data["tra_adj_mat"]
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    G_sub = G.subgraph(subgraph_nodes).copy()
    edges = list(G_sub.edges())
    if not edges:
        raise RuntimeError("Subgraph has no edges.")
    return G_sub, edges


def run_sim(
    *,
    G_sub,
    edges,
    road_length: int,
    num_vehicles: int,
    max_speed: int,
    human_brake_prob: float,
    random_braking_reduction: float,
    penetration_rate: float,
    num_timesteps: int,
    seed: int
):
    random.seed(seed)

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

    cars = {}

    def try_spawn_car(car_id, max_tries=20000):
        for _ in range(max_tries):
            edge = random.choice(edges)
            pos = random.randrange(0, road_length)
            if is_empty(edge, pos):
                # keep RNG call order stable for reproducibility
                speed0 = random.randrange(0, max_speed + 1)
                is_auto = (random.random() < penetration_rate)
                is_human = (not is_auto)

                cars[car_id] = Cars(
                    v_max=max_speed,
                    current_edge=edge,
                    is_human=is_human,
                    lane=0,
                    pos=pos,
                    speed=speed0,
                )
                place(edge, pos, car_id)
                return True
        return False

    spawned = 0
    for cid in range(num_vehicles):
        if try_spawn_car(cid):
            spawned += 1

    def jam_metrics():
        jammed = 0
        clusters = 0
        for e in edges:
            lane = lane_cells(e)
            jam_flags = [False] * road_length

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
        (_, v) = edge
        lane = lane_cells(edge)

        car_ids = [cid for cid in lane if cid is not None]
        if not car_ids:
            return [], 0

        car_ids.sort(key=lambda cid: cars[cid].pos, reverse=True)

        for cid in car_ids:
            remove(edge, cars[cid].pos)

        exits = []
        moved_count = 0

        for idx, cid in enumerate(car_ids):
            car = cars[cid]
            pos = car.pos

            intended_next = None
            can_exit = False

            if pos == road_length - 1:
                intended_next = choose_next_edge(v)
                if intended_next is not None and is_empty(intended_next, 0):
                    can_exit = True

            if idx == 0:
                gap = (road_length - 1) - pos
            else:
                front_cid = car_ids[idx - 1]
                front_pos = cars[front_cid].pos
                gap = (front_pos - pos) - 1

            if pos == road_length - 1 and can_exit:
                gap = 1

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

        for cid, from_edge, _, intended_next in exits:
            car = cars[cid]

            if intended_next is not None and is_empty(intended_next, 0):
                car.edge = intended_next
                car.pos = 0
                car.speed = 0
                place(intended_next, 0, cid)
                entered += 1
            else:
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

    series = {k: [] for k in ["moved", "entered", "queued", "jammed_cells", "jam_clusters"]}

    for _ in range(num_timesteps):
        all_exits = []
        moved_total = 0

        for e in sorted(edges):
            exits, moved = update_edge_nasch(e)
            all_exits.extend(exits)
            moved_total += moved

        entered, queued = process_exits(all_exits)
        jammed, clusters = jam_metrics()

        series["moved"].append(moved_total)
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


def summarize(series, warmup_steps: int):
    def sstats(x):
        x2 = x[warmup_steps:]
        return {
            "mean": stats.mean(x2),
            "sd": stats.pstdev(x2),
            "min": min(x2),
            "max": max(x2),
        }

    summary = {k: sstats(v) for k, v in series.items()}

    flow = series["entered"][warmup_steps:]
    mean_flow = stats.mean(flow)
    sd_flow = stats.pstdev(flow)
    cv = sd_flow / mean_flow if mean_flow != 0 else 0

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

    mse = sum((flow[i] - flow[i - 1]) ** 2 for i in range(1, len(flow))) / (len(flow) - 1)
    nmse = mse / (mean_flow ** 2) if mean_flow != 0 else 0

    summary["predictability"] = {
        "CV": cv,
        "lag1_autocorr": autocorr,
        "naive_MSE": mse,
        "normalized_MSE": nmse,
    }

    return summary
