import scipy.io
import networkx as nx
import matplotlib.pyplot as plt

data = scipy.io.loadmat('traffic_dataset.mat')
train_x = data['tra_X_tr']
adj = data['tra_adj_mat']
starting = data['tra_X_tr'][0,0]
print(train_x)
Graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
for i in range(36):
    Graph.nodes[i]['lanes'] = starting[i, 45]


lane_sizes = [Graph.nodes[i]['lanes'] * 200 for i in Graph.nodes]

plt.figure(figsize=(12, 10))
pos = nx.kamada_kawai_layout(Graph) 


nx.draw(Graph, pos, 
        with_labels=True, 
        node_size=lane_sizes, 
        node_color="skyblue", 
        edge_color="#aaaaaa",
        width=1.5,
        arrowsize=15,
        font_size=9,
        font_weight='bold')

plt.title("Traffic Sensor Network: Node Size by Lane Count (2017)")
plt.show()
