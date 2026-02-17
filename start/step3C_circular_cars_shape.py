import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
from Cars import Cars
import random

# ----------------------------
# Simulation parameters
# ----------------------------
L = 100
num_cars = 20
v_max = 5
p = 0.25
steps = 600
alpha = 0.0   # 0.0 humans, 1.0 autonomous
seed = 1
random.seed(seed)

# ----------------------------
# Visual parameters
# ----------------------------
R = 20.0
interval_ms = 250  # slow it down (try 120–250)

# ----------------------------
# Build ring occupancy + cars
# ----------------------------
lane = [None] * L
cars = []

spacing = L // num_cars
positions = [(i * spacing) % L for i in range(num_cars)]
k_auto = int(round(alpha * num_cars))

for i, pos in enumerate(positions):
    is_human = (i >= k_auto)
    c = Cars(v_max=v_max, current_edge=(0, 0), human=is_human, lane=0, pos=pos)
    cars.append(c)
    lane[pos] = c

def gap_ahead(pos):
    for d in range(1, L):
        j = (pos + d) % L
        if lane[j] is not None:
            return d - 1
    return L - 1

def jam_clusters():
    jam = [0] * L
    for c in cars:
        if c.speed <= 1:
            jam[c.pos] = 1
    if sum(jam) == 0:
        return 0
    clusters = 0
    for i in range(L):
        if jam[i] == 1 and jam[i - 1] == 0:
            clusters += 1
    return clusters

# ----------------------------
# Car silhouette (local coords)
# Pointing "up" in local space; rotation will align to tangent
# ----------------------------
def make_car_polygon():
    pts = np.array([
        [-0.6, -1.0],   # rear left
        [ 0.6, -1.0],   # rear right
        [ 0.7, -0.4],
        [ 0.5,  0.2],
        [ 0.3,  0.7],
        [-0.3,  0.7],
        [-0.5,  0.2],
        [-0.7, -0.4],
    ])
    return Polygon(pts, closed=True)


def pos_to_angle(pos):
    return 2 * np.pi * (pos / L)

def set_car_patch(poly, pos, speed):
    ang = pos_to_angle(pos)

    x = R * np.cos(ang)
    y = R * np.sin(ang)

    # Tangent direction (clockwise motion): angle + pi/2
    rot = ang - np.pi / 2

    poly.set_facecolor("red" if speed <= 1 else "green")
    poly.set_edgecolor("black")
    poly.set_linewidth(0.6)

    # Scale a bit so it looks like a car
    trans = (
        Affine2D()
        .rotate(np.pi/2) 
        .scale(1.0, 0.9)       # tweak silhouette proportions
        .rotate(rot)
        .translate(x, y)
        + ax.transData
    )
    poly.set_transform(trans)

# ----------------------------
# Figure
# ----------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")
ax.set_xlim(-(R + 3), (R + 3))
ax.set_ylim(-(R + 3), (R + 3))
ax.axis("off")
ax.set_title(f"Circular Ring Road (alpha={alpha})", pad=16)

theta = np.linspace(0, 2*np.pi, 400)
ax.plot(R*np.cos(theta), R*np.sin(theta), linewidth=2, alpha=0.35)

info = ax.text(0, R + 2.2, "", ha="center", va="center", fontsize=11)

# Create car patches
patches = []
for _ in cars:
    car_poly = make_car_polygon()
    ax.add_patch(car_poly)
    patches.append(car_poly)

def update(frame):
    # clear occupancy
    for c in cars:
        lane[c.pos] = None

    # update cars
    for c in sorted(cars, key=lambda x: x.pos, reverse=True):
        c.accelarate()
        g = gap_ahead(c.pos)
        c.brake(g)
        if c.isHuman:
            c.randomise(p)
        c.move()
        c.pos %= L
        lane[c.pos] = c

    # update visuals
    for poly, c in zip(patches, cars):
        set_car_patch(poly, c.pos, c.speed)

    slow = sum(1 for c in cars if c.speed <= 1)
    clusters = jam_clusters()
    info.set_text(f"t={frame:04d}   slow_cars={slow}   jam_clusters={clusters}")

    return patches + [info]

ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval_ms, blit=False)
plt.show()
