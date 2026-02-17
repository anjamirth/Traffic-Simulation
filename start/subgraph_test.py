import scipy.io
import networkx as nx
import matplotlib.pyplot as plt

# Load dataset
data = scipy.io.loadmat("traffic_dataset.mat")
adj = data["tra_adj_mat"]

# Build directed graph
G = nx.from_numpy_array(adj, create_using=nx.DiGraph)

# Select hub-based subgraph
subgraph_nodes = [25, 0, 4, 14, 21, 22, 24, 29]
G_sub = G.subgraph(subgraph_nodes).copy()

print("Nodes:", G_sub.nodes())
print("Edges:", G_sub.edges())
print("Number of nodes:", G_sub.number_of_nodes())
print("Number of edges:", G_sub.number_of_edges())

# Plot
plt.figure(figsize=(7,6))
pos = nx.spring_layout(G_sub, seed=42)

nx.draw(
    G_sub,
    pos,
    with_labels=True,
    node_size=900,
    node_color="skyblue",
    arrows=True
)

plt.title("8-Node Traffic Subgraph")
plt.show()
