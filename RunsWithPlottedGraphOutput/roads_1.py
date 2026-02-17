import scipy.io
import networkx as nx

# ----------------------------
# Settings (locked)
# ----------------------------
L = 100  # cells per road
subgraph_nodes = [25, 0, 4, 14, 21, 22, 24, 29]

# ----------------------------
# Load dataset + build graph
# ----------------------------
data = scipy.io.loadmat("traffic_dataset.mat")
adj = data["tra_adj_mat"]
G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
G_sub = G.subgraph(subgraph_nodes).copy()

# ----------------------------
# Lane counts from features (same logic as your earlier code)
# ----------------------------
starting = data["tra_X_tr"][0, 0]  # first timestep (likely 36 x 48)

try:
    starting = starting.toarray()
except Exception:
    pass

def lanes_for_node(n: int) -> int:
    raw = float(starting[n, 45])  # same column you used before :contentReference[oaicite:1]{index=1}
    lanes = int(round(raw))
    if lanes < 1:
        lanes = 1
    return lanes

# ----------------------------
# Build "roads": each directed edge (u,v) has lane_count lanes,
# each lane is a list of L empty cells (None)
# ----------------------------
roads = {}  # (u,v) -> list_of_lanes; lane is [None]*L

for (u, v) in G_sub.edges():
    lane_count = min(lanes_for_node(u), lanes_for_node(v))  # conservative capacity
    roads[(u, v)] = [[None] * L for _ in range(lane_count)]

# ----------------------------
# Sanity summary
# ----------------------------
print("Subgraph nodes:", list(G_sub.nodes()))
print("Subgraph edges:", list(G_sub.edges()))
print("\nRoad storage created:")
for (u, v), lanes in roads.items():
    print(f"  edge ({u}->{v}): {len(lanes)} lane(s), {len(lanes[0])} cells per lane")

print(f"\nTotal directed edges stored: {len(roads)}")
