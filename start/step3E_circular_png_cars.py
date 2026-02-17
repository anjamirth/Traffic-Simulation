import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.transforms import Affine2D
from Cars import Cars

# ----------------------------
# 1) Simulation parameters
# ----------------------------
L = 100
num_cars = 20
v_max = 5
p = 0.25
steps = 600
alpha = 0.0
seed = 1

# Reproducible randomness (IMPORTANT)
random.seed(seed)
np.random.seed(seed)

# ----------------------------
# 2) Visual parameters
# ----------------------------
R = 10.0
interval_ms = 140
car_size = 0.4
radial_nudge = -0.35

# ----------------------------
# 2a) Load + normalise car PNG (IMPORTANT)
# ----------------------------
car_img = mpimg.imread("car-png-16843.png")

def to_float01(img):
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img

def ensure_rgba(img):
    if img.shape[2] == 4:
        return img
    a = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
    return np.concatenate([img, a], axis=2)

car_img = ensure_rgba(to_float01(car_img))

def whiten(img, strength=0.35):
    out = img.copy()
    out[..., :3] = np.clip((1 - strength) * out[..., :3] + strength * 1.0, 0, 1)
    return out

def red_tint(img, strength=0.75):
    out = img.copy()
    rgb = out[..., :3]
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0
    out[..., :3] = np.clip((1 - strength) * rgb + strength * red, 0, 1)
    return out

# Make red REALLY obvious by whitening first
car_img_normal = whiten(car_img, strength=0.35)
car_img_jam = red_tint(car_img_normal, strength=0.80)

# ----------------------------
# 3) Build ring
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

def jam_clusters(threshold=1):
    jam = [0] * L
    for c in cars:
        if c.speed <= threshold:
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

# ----------------------------
# 4) Figure
# ----------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")
ax.set_xlim(-(R + 3), (R + 3))
ax.set_ylim(-(R + 3), (R + 3))
ax.axis("off")
ax.set_title(f"Circular Ring Road (alpha={alpha}, p={p}, seed={seed})", pad=16)

theta = np.linspace(0, 2*np.pi, 400)
ax.plot(R * np.cos(theta), R * np.sin(theta), linewidth=3, alpha=0.20)

info = ax.text(0, R + 2.2, "", ha="center", va="center", fontsize=11)

# One AxesImage per car
images = []
for _ in cars:
    im = ax.imshow(car_img_normal, extent=[-car_size, car_size, -car_size, car_size], zorder=3)
    images.append(im)

def set_image(im, x, y, rot_rad):
    im.set_extent([x - car_size, x + car_size, y - car_size, y + car_size])
    im.set_transform(Affine2D().rotate_around(x, y, rot_rad) + ax.transData)

def update(frame):
    # clear occupancy
    for c in cars:
        lane[c.pos] = None

    # update cars (fixed deterministic order)
    for c in sorted(cars, key=lambda x: x.pos, reverse=True):
        c.accelarate()
        g = gap_ahead(c.pos)
        c.brake(g)

        if c.isHuman:
            c.randomise(p)

        c.move()
        c.pos %= L
        lane[c.pos] = c

    # visuals
    jam_threshold = 1   # if you want more obvious jams, set to 2
    for im, c in zip(images, cars):
        ang = pos_to_angle(c.pos)

        x = R * np.cos(ang) + radial_nudge * np.cos(ang)
        y = R * np.sin(ang) + radial_nudge * np.sin(ang)

        # Tangent direction for increasing pos
        rot = ang + np.pi / 2

        im.set_data(car_img_jam if c.speed <= jam_threshold else car_img_normal)
        set_image(im, x, y, rot)

    slow = sum(1 for c in cars if c.speed <= jam_threshold)
    clusters = jam_clusters(threshold=jam_threshold)
    info.set_text(f"t={frame:04d}   slow_cars={slow}   jam_clusters={clusters}")

    return images + [info]

ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval_ms, blit=False)
plt.show()
