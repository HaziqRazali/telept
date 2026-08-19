"""Mocap 3D/2D rendering for the GUI (matplotlib -> numpy image)."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from config import MARKER_NAMES

_MARKER_COLORS = {
    "Board1": "red", "Board2": "red", "Board3": "red",
    "Board4": "blue", "Board5": "blue", "Board6": "blue",
    "*": "gray",
}


def _marker_color(name: str) -> str:
    for key, col in _MARKER_COLORS.items():
        if key in name:
            return col
    return "black"


def fig_to_img(fig) -> np.ndarray:
    """Render a matplotlib figure to an RGB numpy image and close it."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def draw_box_3d(ax, lo: np.ndarray, hi: np.ndarray, color="lime"):
    """Draw an axis-aligned wireframe box given lo/hi corners."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    # 8 corners
    corners = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                        [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                        [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    segs = [[corners[a], corners[b]] for a, b in edges]
    ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=2))


def render_mocap_3d(
    mocap: dict,
    frame_idx: int,
    box: tuple[np.ndarray, np.ndarray] | None = None,
    highlight_name: str | None = None,
) -> np.ndarray:
    """3D scatter of all tracked markers at a mocap frame -> RGB image."""
    labels = mocap["labels"]
    xyz = mocap["xyz"]
    pres = mocap["presence"][:, frame_idx]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    for i, name in enumerate(labels):
        if not pres[i]:
            continue
        p = xyz[i, :, frame_idx]
        c = "yellow" if name == highlight_name else _marker_color(name)
        s = 90 if name == highlight_name else 40
        ax.scatter(p[0], p[1], p[2], c=c, s=s, label=name, depthshade=False)

    if box is not None:
        draw_box_3d(ax, box[0], box[1])

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"frame {frame_idx} ({frame_idx / mocap['fps']:.2f}s)")
    if len(ax.collections) or len(ax.lines):
        pass
    return fig_to_img(fig)


def render_mocap_2d(
    mocap: dict,
    frame_idx: int,
    plane: str = "top",
    box: tuple[np.ndarray, np.ndarray] | None = None,
    highlight_name: str | None = None,
) -> np.ndarray:
    """2D projection of the markers (top / front / side) -> RGB image."""
    labels = mocap["labels"]
    xyz = mocap["xyz"]
    pres = mocap["presence"][:, frame_idx]

    pairs = {"top": (0, 1), "front": (0, 2), "side": (1, 2)}
    a, b = pairs[plane]

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, name in enumerate(labels):
        if not pres[i]:
            continue
        p = xyz[i, :, frame_idx]
        c = "yellow" if name == highlight_name else _marker_color(name)
        ax.scatter(p[a], p[b], c=c, s=25, label=name)

    if box is not None:
        lo, hi = box
        xs = [lo[a], hi[a], hi[a], lo[a], lo[a]]
        ys = [lo[b], lo[b], hi[b], hi[b], lo[b]]
        ax.plot(xs, ys, color="lime", lw=2)

    ax.set_xlabel(["X (mm)", "X (mm)", "Y (mm)"][["top", "front", "side"].index(plane)])
    ax.set_ylabel(["Y (mm)", "Z (mm)", "Z (mm)"][["top", "front", "side"].index(plane)])
    ax.set_title(f"{plane} view - frame {frame_idx}")
    ax.set_aspect("equal", adjustable="datalim")
    return fig_to_img(fig)


def render_mocap_overview(mocap: dict, box=None) -> np.ndarray:
    """Full-recording overview: all marker trajectories (3D) with the box."""
    labels = mocap["labels"]
    xyz = mocap["xyz"]
    pres = mocap["presence"]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i, name in enumerate(labels):
        p = xyz[i, :, pres[i]]
        if p.shape[1] < 5:
            continue
        ax.plot(p[0], p[1], p[2], lw=0.6, alpha=0.7, label=name)
    if box is not None:
        draw_box_3d(ax, box[0], box[1])
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("Full recording overview (marker trajectories)")
    return fig_to_img(fig)
