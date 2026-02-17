import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Cars import Cars
import random

# ----------------------------
# Parameters
# ----------------------------
L = 100
num_cars = 20
v_max = 5
p = 0.25
steps = 300

alpha = 0.0   # change to 1.0 to see autonomous effect

# ----------------------------
# Setup ring
# ----------------------------
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

# ----------------------------
# Animation setup
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 2))
ax.set_xlim(0, L)
ax.set_ylim(-1, 1)
ax.set_yticks([])
ax.set_title(f"Ring Road Simulation (alpha={alpha})")

scat = ax.scatter([], [], s=100)

def update(frame):
    global lane

    # Remove cars
    for c in cars:
        lane[c.pos] = None

    # Update cars
    for c in sorted(cars, key=lambda x: x.pos, reverse=True):
        c.accelarate()
        gap = gap_ahead(c.pos)
        c.brake(gap)

        if c.isHuman:
            c.randomise(p)

        c.move()
        c.pos %= L
        lane[c.pos] = c

    # Prepare scatter data
    xs = [c.pos for c in cars]
    ys = [0] * len(cars)

    colors = ["red" if c.speed <= 1 else "green" for c in cars]

    scat.set_offsets(list(zip(xs, ys)))
    scat.set_color(colors)

    return scat,

ani = animation.FuncAnimation(fig, update, frames=steps, interval=50)
plt.show()
