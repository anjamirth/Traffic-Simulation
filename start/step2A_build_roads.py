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
# Get lane counts from features
# Dataset says "number of lanes" is a feature. :contentReference[oaicite:0]{index=0}
# Your earlier code uses column 45. :contentReference[oaicite:1]{index=1}
# ----------------------------
starting = data["tra_X_tr"][0, 0]  # first timestep (36 x 48)
# Some .mat exports store this as sparse; make it dense if needed
try:
    starting_dense = starting.toarray()
except Exception:
    starting_dense = starting

def lanes_for_node(n: int) -> int:
    # Column 45 is where you previously read lanes.
    raw = float(starting_dense[n, 45])
    # Safety: ensure at least 1 lane
    lanes = int(round(raw)) if raw >= 1 else 1
    return lanes

# ----------------------------
# Build "roads": for each directed edge (u,v),
# create lanes lanes, each lane is a list of L empty cells (None)
# ----------------------------
roads = {}  # (u,v) -> list_of_lanes, where each lane is [None]*L

for (u, v) in G_sub.edges():
    lane_count = min(lanes_for_node(u), lanes_for_node(v))  # conservative capacity
    roads[(u, v)] = [[None] * L for _ in range(lane_count)]

# ----------------------------
# Print a sanity summary
# ----------------------------
print("Subgraph nodes:", list(G_sub.nodes()))
print("Subgraph edges:", list(G_sub.edges()))
print("\nRoad storage created:")
for (u, v), lanes in roads.items():
    print(f"  edge ({u}->{v}): {len(lanes)} lane(s), {len(lanes[0])} cells per lane")

print(f"\nTotal directed edges stored: {len(roads)}")
