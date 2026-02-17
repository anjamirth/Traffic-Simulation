print("=== Step 2D starting ===")

from Cars import Cars
import scipy.io
import networkx as nx

L = 100
subgraph_nodes = [25, 0, 4, 14, 21, 22, 24, 29]

p = 0.2
steps = 25
test_edge = (25, 22)
lane = 0

data = scipy.io.loadmat("traffic_dataset.mat")
adj = data["tra_adj_mat"]
G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
G_sub = G.subgraph(subgraph_nodes).copy()

roads = {(u, v): [[None] * L] for (u, v) in G_sub.edges()}

# Two human cars: carA behind carB
carA = Cars(v_max=5, current_edge=test_edge, human=True, lane=lane, pos=0)
carB = Cars(v_max=5, current_edge=test_edge, human=True, lane=lane, pos=12)

roads[test_edge][lane][carA.pos] = carA
roads[test_edge][lane][carB.pos] = carB

def find_gap(car, lane_cells):
    """
    Gap = number of empty cells in front until the next car.
    If next car at position j, gap = (j - car.pos - 1)
    If no car ahead, gap = max possible (remaining road).
    """
    for j in range(car.pos + 1, L):
        if lane_cells[j] is not None:
            return j - car.pos - 1
    return (L - 1) - car.pos

def print_lane(t):
    view_len = 50
    lane_view = roads[test_edge][lane][:view_len]
    s = "".join(
        "A" if cell is carA else ("B" if cell is carB else ".")
        for cell in lane_view
    )
    print(f"t={t:02d} A(pos={carA.pos:02d},v={carA.speed})  B(pos={carB.pos:02d},v={carB.speed})  {s}")

print_lane(0)

for t in range(1, steps + 1):
    lane_cells = roads[test_edge][lane]

    # Remove both cars from old cells
    lane_cells[carA.pos] = None
    lane_cells[carB.pos] = None

    # Update B first (front car)
    carB.accelarate()
    gapB = find_gap(carB, lane_cells)
    carB.brake(gapB)
    carB.randomise(p)
    carB.move()

    if carB.pos >= L:
        carB.pos = L - 1
        carB.speed = 0

    # Place B back so A can "see" it
    lane_cells[carB.pos] = carB

    # Update A (rear car)
    carA.accelarate()
    gapA = find_gap(carA, lane_cells)
    carA.brake(gapA)
    carA.randomise(p)
    carA.move()

    if carA.pos >= L:
        carA.pos = L - 1
        carA.speed = 0

    # Place A back
    lane_cells[carA.pos] = carA

    print_lane(t)

print("=== Step 2D done ===")
