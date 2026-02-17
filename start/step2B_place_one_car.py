print("=== Step 2B starting ===")

try:
    from Cars import Cars
    print("Imported Cars class OK")
except Exception as e:
    print("FAILED importing Cars:", repr(e))
    raise

import scipy.io
import networkx as nx

print("Loaded imports OK")

L = 100
subgraph_nodes = [25, 0, 4, 14, 21, 22, 24, 29]

data = scipy.io.loadmat("traffic_dataset.mat")
adj = data["tra_adj_mat"]
print("Loaded .mat OK")

G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
G_sub = G.subgraph(subgraph_nodes).copy()
print("Built subgraph OK. Edges:", G_sub.number_of_edges())

roads = {(u, v): [[None] * L] for (u, v) in G_sub.edges()}
print("Built roads dict OK. Stored edges:", len(roads))

test_edge = (25, 22)
print("Test edge exists in roads?", test_edge in roads)

# If it's not present, pick the first edge that exists
if test_edge not in roads:
    test_edge = next(iter(roads.keys()))
    print("Using fallback edge:", test_edge)

lane = 0
start_pos = 0

car0 = Cars(v_max=5, current_edge=test_edge, human=True, lane=lane, pos=start_pos)
roads[test_edge][lane][start_pos] = car0

view_len = 30
lane_view = roads[test_edge][lane][:view_len]

print("Lane view:", "".join("C" if cell is not None else "." for cell in lane_view))
print("Car pos:", car0.pos, "speed:", car0.speed, "edge:", car0.edge)

print("=== Step 2B done ===")
