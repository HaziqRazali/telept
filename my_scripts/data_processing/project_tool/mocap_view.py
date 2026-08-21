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


def fixed_range(mocap: dict, pad_mm: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Global axis range (lo, hi, each (3,)) over ALL visible markers in the
    whole recording, padded by ``pad_mm``.

    The result is cached on the mocap dict so the scrub view keeps a stable
    scale instead of re-fitting to each frame's markers (which makes the
    axes jump around).
    """
    key = "_fixed_range_mm"
    cached = mocap.get(key)
    if cached is not None:
        return cached
    xyz = mocap["xyz"]          # (N, 3, F)
    pres = mocap["presence"]    # (N, F)
    seen = []
    for i in range(xyz.shape[0]):
        m = pres[i]
        if m.any():
            # NB: xyz[i] is (3, F); mask the LAST axis to keep (3, n).
            # (xyz[i, :, m] reorders axes and would give (n, 3) -- wrong.)
            seen.append(xyz[i][:, m])
    if seen:
        P = np.concatenate(seen, axis=1)      # (3, M)
        lo = P.min(axis=1) - pad_mm
        hi = P.max(axis=1) + pad_mm
    else:
        lo = -np.ones(3) * 1000.0
        hi = np.ones(3) * 1000.0
    mocap[key] = (lo, hi)
    return lo, hi


def render_mocap_3d(
    mocap: dict,
    frame_idx: int,
    box: tuple[np.ndarray, np.ndarray] | None = None,
    highlight_name: str | None = None,
    rng: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """3D scatter of all tracked markers at a mocap frame -> RGB image.

    Axes are locked to the full-recording range (fixed_range) so the view
    doesn't rescale between frames; pass ``rng`` to override.
    """
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

    lo, hi = fixed_range(mocap) if rng is None else rng
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])

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
    rng: tuple[np.ndarray, np.ndarray] | None = None,
    return_transform: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """2D projection of the markers (top / front / side) -> RGB image.

    The window is locked to the full-recording range (fixed_range), made
    square so the equal-aspect projection stays exact and never rescales
    between frames.  Pass ``rng`` to override.

    If ``return_transform`` is True, returns ``(img, info)`` where ``info``
    maps image-pixel coordinates to data (mm) coordinates (for click-to-draw):
    ``info = {plane, xlim, ylim, bbox_img}`` with ``bbox_img`` = axes bbox in
    image pixels (origin top-left).
    """
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

    # fixed, square window (equal aspect stays exact and stable)
    glo, ghi = fixed_range(mocap) if rng is None else rng
    half = max(ghi[a] - glo[a], ghi[b] - glo[b]) / 2
    cx = (glo[a] + ghi[a]) / 2
    cy = (glo[b] + ghi[b]) / 2
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    ax.set_xlabel(["X (mm)", "X (mm)", "Y (mm)"][["top", "front", "side"].index(plane)])
    ax.set_ylabel(["Y (mm)", "Z (mm)", "Z (mm)"][["top", "front", "side"].index(plane)])
    ax.set_title(f"{plane} view - frame {frame_idx}")
    ax.set_aspect("equal", adjustable="box")

    info = None
    if return_transform:
        fig.canvas.draw()
        bb = ax.get_window_extent(fig.canvas.get_renderer())
        w_in, h_in = fig.get_size_inches()
        fig_px_h = int(h_in * fig.dpi)
        # display bbox has origin bottom-left; image (buffer_rgba) has top-left
        info = {
            "plane": plane,
            "xlim": (float(ax.get_xlim()[0]), float(ax.get_xlim()[1])),
            "ylim": (float(ax.get_ylim()[0]), float(ax.get_ylim()[1])),
            "bbox_img": (float(bb.x0), float(fig_px_h - bb.y1),
                          float(bb.width), float(bb.height)),
        }
    img = fig_to_img(fig)
    return (img, info) if return_transform else img


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
