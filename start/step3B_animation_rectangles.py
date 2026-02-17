import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from Cars import Cars
import random

L = 100
num_cars = 20
v_max = 5
p = 0.25
steps = 400
alpha = 0.0

lane = [None] * L
cars = []

spacing = L // num_cars
positions = [(i * spacing) % L for i in range(num_cars)]
k_auto = int(round(alpha * num_cars))

for i, pos in enumerate(positions):
    is_human = (i >= k_auto)
    c = Cars(v_max=v_max, current_edge=(0,0), human=is_human, lane=0, pos=pos)
    cars.append(c)
    lane[pos] = c

def gap_ahead(pos):
    for d in range(1, L):
        j = (pos + d) % L
        if lane[j] is not None:
            return d - 1
    return L - 1

fig, ax = plt.subplots(figsize=(12, 2))
ax.set_xlim(0, L)
ax.set_ylim(-1, 1)
ax.set_yticks([])
ax.set_title(f"Ring Road Simulation (alpha={alpha})")

patches = []

for _ in cars:
    rect = Rectangle((0, -0.2), 2, 0.4)
    ax.add_patch(rect)
    patches.append(rect)

def update(frame):
    global lane

    # remove cars from lane
    for c in cars:
        lane[c.pos] = None

    # update cars
    for c in sorted(cars, key=lambda x: x.pos, reverse=True):
        c.accelarate()
        gap = gap_ahead(c.pos)
        c.brake(gap)

        if c.isHuman:
            c.randomise(p)

        c.move()
        c.pos %= L
        lane[c.pos] = c

    # update visuals
    for rect, c in zip(patches, cars):
        rect.set_x(c.pos)
        rect.set_y(-0.2)
        rect.set_facecolor("red" if c.speed <= 1 else "green")

    return patches

ani = animation.FuncAnimation(fig, update, frames=steps, interval=40)
plt.show()
