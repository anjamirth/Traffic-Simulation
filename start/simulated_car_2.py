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
seed = 10

# Reproducible randomness
random.seed(seed)
np.random.seed(seed)

# ----------------------------
# 2) Visual parameters
# ----------------------------
R = 10.0
interval_ms = 140
radial_nudge = -0.35  # inward shift so the visible car sits on the road

# ----------------------------
# 2a) Load + normalise car PNG
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

def red_tint(img, strength=0.80):
    out = img.copy()
    rgb = out[..., :3]
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0
    out[..., :3] = np.clip((1 - strength) * rgb + strength * red, 0, 1)
    return out

car_img_normal = whiten(car_img, strength=0.35)
car_img_jam = red_tint(car_img_normal, strength=0.80)

# ----------------------------
# 2b) Size cars based on cell arc-length (prevents overlap when adjacent)
# ----------------------------
cell_arc = 2 * np.pi * R / L   # arc length of 1 cell in plot units

car_height = 0.55 * cell_arc   # across-road thickness (try 0.35–0.55)
car_width  = 0.95 * cell_arc   # along-road footprint (try 0.70–1.00)

# Keep original image aspect ratio by adjusting width from height, but cap it
img_h, img_w = car_img_normal.shape[0], car_img_normal.shape[1]
aspect = img_w / img_h
car_width = min(car_width, car_height * aspect)

# Optional: road "band" jitter to stop visual stacking in jams (no flicker)
lane_band = 0.12  # try 0.08–0.18

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

def gap_ahead(pos, lane_snapshot):
    for d in range(1, L):
        j = (pos + d) % L
        if lane_snapshot[j] is not None:
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
    im = ax.imshow(
        car_img_normal,
        extent=[-car_width/2, car_width/2, -car_height/2, car_height/2],
        zorder=3
    )
    images.append(im)

def set_image(im, x, y, rot_rad):
    im.set_extent([x - car_width/2, x + car_width/2,
                   y - car_height/2, y + car_height/2])
    im.set_transform(Affine2D().rotate_around(x, y, rot_rad) + ax.transData)

def update(frame):
    # lane snapshot at start of timestep
    lane_old = [None] * L
    for c in cars:
        lane_old[c.pos] = c

    # ---- Phase 1: compute new speeds using lane_old ----
    for c in cars:
        c.accelarate()
        g = gap_ahead(c.pos, lane_old)
        c.brake(g)
        if c.isHuman:
            c.randomise(p)

    # ---- Phase 2: move simultaneously ----
    new_positions = [(c.pos + c.speed) % L for c in cars]

    # collision check
    if len(new_positions) != len(set(new_positions)):
        # If this triggers, your Cars.move/brake logic allows illegal jumps.
        # Print a helpful debug and stop.
        raise AssertionError("Collision: two cars target same cell in parallel update")

    # apply movement + build lane for next frame
    lane_new = [None] * L
    for c, new_pos in zip(cars, new_positions):
        c.pos = new_pos
        lane_new[c.pos] = c

    # update global lane to lane_new so visual + next frame match
    global lane
    lane = lane_new

    # ---- visuals (unchanged) ----
    jam_threshold = 1
    for im, c in zip(images, cars):
        ang = pos_to_angle(c.pos)

        x = R * np.cos(ang) + radial_nudge * np.cos(ang)
        y = R * np.sin(ang) + radial_nudge * np.sin(ang)

        tx, ty = -np.sin(ang), np.cos(ang)
        j = (hash(id(c)) % 1000) / 1000.0
        j = (j - 0.5) * 2.0
        x += lane_band * j * tx
        y += lane_band * j * ty

        rot = ang + np.pi / 2

        im.set_data(car_img_jam if c.speed <= jam_threshold else car_img_normal)
        set_image(im, x, y, rot)

    slow = sum(1 for c in cars if c.speed <= jam_threshold)
    clusters = jam_clusters(threshold=jam_threshold)
    info.set_text(f"t={frame:04d}   slow_cars={slow}   jam_clusters={clusters}")

    return images + [info]

ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval_ms, blit=False)
plt.show()
