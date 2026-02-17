# ring_visual.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.transforms import Affine2D

from ring_model import RingRoadModel

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

def green_tint(img, strength=0.65):
    out = img.copy().astype(np.float32)
    rgb = out[..., :3]

    green = np.zeros_like(rgb)
    green[..., 1] = 1.0  # pure green overlay

    out[..., :3] = np.clip((1 - strength) * rgb + strength * green, 0, 1)
    return out.astype(np.float32)


def red_tint(img, strength=0.80):
    out = img.copy()
    rgb = out[..., :3]
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0
    out[..., :3] = np.clip((1 - strength) * rgb + strength * red, 0, 1)
    return out

def blue_tint(img, strength=0.55):
    out = img.copy()
    rgb = out[..., :3]
    blue = np.zeros_like(rgb)
    blue[..., 2] = 1.0
    out[..., :3] = np.clip((1 - strength) * rgb + strength * blue, 0, 1)
    return out

def animate_ring(
    L=100, num_cars=20, v_max=5, p=0.25, steps=600, alpha=0.5, seed=10,
    R=10.0, interval_ms=140, radial_nudge=-0.35, lane_band=0.12,
    car_png_path="car-png-16843.png", jam_threshold=1
):
    # Build model
    model = RingRoadModel(L=L, num_cars=num_cars, v_max=v_max, p=p, alpha=alpha, seed=seed)

    # Load car PNG
    car_img = ensure_rgba(to_float01(mpimg.imread(car_png_path)))
    car_img_normal = green_tint(car_img, strength=0.85)
    car_img_jam = red_tint(car_img_normal, strength=0.80)
    car_img_auto = blue_tint(car_img_normal, strength=0.55)

    # Size cars from cell arc length
    cell_arc = 2 * np.pi * R / L
    car_h = 0.45 * cell_arc
    car_w = 0.90 * cell_arc

    img_h, img_w = car_img_normal.shape[0], car_img_normal.shape[1]
    aspect = img_w / img_h
    car_w = min(car_w, car_h * aspect)

    def pos_to_angle(pos):
        return 2 * np.pi * (pos / L)

    # Figure
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-(R + 3), (R + 3))
    ax.set_ylim(-(R + 3), (R + 3))
    ax.axis("off")
    ax.set_title(f"Circular Ring Road (alpha={alpha}, p={p}, seed={seed})", pad=16)

    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(R * np.cos(theta), R * np.sin(theta), linewidth=1, alpha=0.20)

    info = ax.text(0, R + 2.2, "", ha="center", va="center", fontsize=11)
    ax.text(-(R+2.6), -(R+2.2), "Human", fontsize=8, color="green")
    ax.text(-(R+2.6), -(R+2.7), "Autonomous", fontsize=8, color="blue")
    ax.text(-(R+2.6), -(R+3.2), "Slow/Jam", fontsize=8, color="red")


    # one image per car
    images = []
    for _ in range(num_cars):
        im = ax.imshow(
            car_img_normal,
            extent=[-car_w/2, car_w/2, -car_h/2, car_h/2],
            zorder=3
        )
        images.append(im)

    def set_image(im, x, y, rot_rad):
        im.set_extent([x - car_w/2, x + car_w/2, y - car_h/2, y + car_h/2])
        im.set_transform(Affine2D().rotate_around(x, y, rot_rad) + ax.transData)

    # stable jitter per car index
    jitter = np.linspace(-1, 1, num_cars)

    def update(frame):
        stats = model.step(jam_threshold=jam_threshold)
        car_state = model.get_positions_speeds()  # (pos, speed, isHuman)

        for idx, (im, (pos, speed, _isHuman)) in enumerate(zip(images, car_state)):
            ang = pos_to_angle(pos)

            x = R * np.cos(ang) + radial_nudge * np.cos(ang)
            y = R * np.sin(ang) + radial_nudge * np.sin(ang)

            # tangent jitter so jam stacks are readable
            tx, ty = -np.sin(ang), np.cos(ang)
            x += lane_band * jitter[idx] * tx
            y += lane_band * jitter[idx] * ty

            rot = ang + np.pi / 2  # tangent direction

            if speed <= jam_threshold:
                im.set_data(car_img_jam)          # jam overrides everything
            else:
                im.set_data(car_img_normal if _isHuman else car_img_auto)

            set_image(im, x, y, rot)

        info.set_text(f"t={stats.t:04d}   slow_cars={stats.slow_cars}   jam_clusters={stats.jam_clusters}")
        return images + [info]

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval_ms, blit=False)
    plt.show()


if __name__ == "__main__":
    animate_ring()
