import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Cars import Cars
import random

L = 100
num_cars = 20
v_max = 5
p = 0.25
steps = 500
alpha = 0.0
seed = 1
random.seed(seed)

R = 10.0
interval_ms = 140

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

def pos_to_angle(pos):
    return 2 * np.pi * (pos / L)

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")
ax.set_xlim(-(R + 3), (R + 3))
ax.set_ylim(-(R + 3), (R + 3))
ax.axis("off")
ax.set_title(f"Circular Ring Road (alpha={alpha})", pad=16)

theta = np.linspace(0, 2*np.pi, 400)
ax.plot(R*np.cos(theta), R*np.sin(theta), linewidth=2, alpha=0.35)

info = ax.text(0, R + 2.2, "", ha="center", va="center", fontsize=11)

# create one text object per car (we move them)
texts = []
for i in range(len(cars)):
    t = ax.text(0, 0, '🌕', fontname='Segoe UI Emoji', fontsize=18, ha="center", va="center")
    texts.append(t)

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
    for txt, c in zip(texts, cars):
        ang = pos_to_angle(c.pos)
        x = R * np.cos(ang)
        y = R * np.sin(ang)

        txt.set_position((x, y))
        txt.set_text("🌕" if not c.isHuman else "🌗")  # optional: different emoji types
        txt.set_color("red" if c.speed <= 1 else "green")

    slow = sum(1 for c in cars if c.speed <= 1)
    clusters = jam_clusters()
    info.set_text(f"t={frame:04d}   slow_cars={slow}   jam_clusters={clusters}")

    return texts + [info]

ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval_ms, blit=False)
plt.show()
