"""Pose-following joint and bone overlays for the Python MHR renderers.

The Unreal project contains static line/joint meshes (``rt_skel``,
``lf_skel``, ``rt_jnts`` and ``lf_jnts``).  Those assets cannot follow a new
pose by themselves.  This module recreates the visible overlay in screen
space from the current MHR joint coordinates instead:

* every bone endpoint is read from the current ``(N, 3)`` MHR joint array;
* each endpoint is projected with the same body-aligned ``ViewSpec`` as the
  mesh; and
* the line and joint marker are redrawn for every rendered frame.

The colours are sRGB approximations of the supplied Unreal joint material
instance colours.  The overlay does not modify MHR vertices or topology.
"""

from __future__ import annotations

import cv2
import numpy as np

from .mhr_shoulder import MHR_BODY_EDGES


POSE_OVERLAY_CHOICES = ("none", "unreal", "legacy")

# OpenCV uses BGR tuples.  These are sRGB display values obtained from the
# linear BaseColor overrides found in the Unreal material instances:
#
#   MI_JointsFresnel_Rt1  -> linear (0.65625, 0.138541, 0.086238)
#   MI_JointsFresnel_Lf1 -> linear (0.234914, 0.424025, 0.828125)
#   MI_JointsFresnel_Skel -> linear (0.263643, 0.848958, 0.220087)
UNREAL_RIGHT_COLOR = (83, 104, 212)     # BGR; red anatomical-right chain
UNREAL_LEFT_COLOR = (235, 174, 133)    # BGR; blue anatomical-left chain
UNREAL_SKELETON_COLOR = (129, 237, 140) # BGR; green torso/central chain
UNREAL_JOINT_COLOR = (235, 235, 235)

# Keep the prior Python renderer appearance available for callers that do not
# request the Unreal-style colours explicitly.
LEGACY_SKELETON_COLOR = (105, 115, 130)
LEGACY_JOINT_COLOR = (55, 75, 95)

_RIGHT_EDGES = frozenset(
    {
        (1, 18), (18, 19), (19, 20), (20, 24),
        (37, 38), (38, 39), (39, 40), (40, 42),
    }
)
_LEFT_EDGES = frozenset(
    {
        (1, 2), (2, 3), (3, 4), (4, 8),
        (37, 74), (74, 75), (75, 76), (76, 78),
    }
)


def _canonical_edge(a: int, b: int) -> tuple[int, int]:
    """Return an edge in the same orientation-independent form every time."""

    return (a, b) if a < b else (b, a)


_RIGHT_EDGES = frozenset(_canonical_edge(*edge) for edge in _RIGHT_EDGES)
_LEFT_EDGES = frozenset(_canonical_edge(*edge) for edge in _LEFT_EDGES)


def _validate_inputs(image: np.ndarray, spec, joints: np.ndarray) -> np.ndarray:
    """Validate and normalize the inputs used by :func:`draw_pose_overlay`."""

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError(f"joints must have shape (N, 3), got {joints.shape}")
    if len(joints) == 0 or not np.isfinite(joints).all():
        raise ValueError("joints must be non-empty and finite")
    if not hasattr(spec, "project"):
        raise TypeError("spec must provide a project(points) method")
    return joints


def draw_pose_overlay(
    image: np.ndarray,
    spec,
    joints: np.ndarray,
    style: str = "unreal",
    *,
    line_thickness: int = 2,
    joint_radius: int = 5,
    draw_joints: bool = True,
    joint_indices=None,
) -> np.ndarray:
    """Draw a pose-following skeleton over a rendered MHR panel.

    Parameters
    ----------
    image:
        BGR image modified in place and returned for convenient chaining.
    spec:
        The renderer's body-aligned view specification.  Its ``project``
        method must map MHR world points to panel pixels.
    joints:
        Current posed MHR joint coordinates, usually shape ``(127, 3)``.
    style:
        ``"unreal"`` uses the red/blue/green Unreal-style colours;
        ``"legacy"`` uses the original neutral Python renderer colours;
        ``"none"`` leaves the image unchanged.
    joint_indices:
        Optional iterable of joint indices to mark.  The lines always use all
        ``MHR_BODY_EDGES``; omitting this argument marks every line endpoint.

    The line endpoints are projected directly from the supplied pose.  No
    rest-pose line mesh, skeletal animation asset, or Unreal runtime is used.
    """

    if style not in POSE_OVERLAY_CHOICES:
        choices = ", ".join(POSE_OVERLAY_CHOICES)
        raise ValueError(f"Unknown pose overlay style {style!r}; choose {choices}.")
    if style == "none":
        return image

    joints = _validate_inputs(image, spec, joints)
    max_index = max(max(edge) for edge in MHR_BODY_EDGES)
    if max_index >= len(joints):
        raise ValueError(
            f"MHR pose overlay needs joint index {max_index}, but only "
            f"{len(joints)} joints were provided."
        )

    projected = np.asarray(spec.project(joints), dtype=np.int32)
    if projected.shape != joints.shape[:1] + (2,):
        raise ValueError(
            f"spec.project(joints) must have shape ({len(joints)}, 2), got {projected.shape}"
        )

    if style == "unreal":
        default_line_color = UNREAL_SKELETON_COLOR
        right_color = UNREAL_RIGHT_COLOR
        left_color = UNREAL_LEFT_COLOR
        joint_color = UNREAL_JOINT_COLOR
    else:
        default_line_color = LEGACY_SKELETON_COLOR
        right_color = default_line_color
        left_color = default_line_color
        joint_color = LEGACY_JOINT_COLOR

    line_thickness = max(1, int(line_thickness))
    joint_radius = max(1, int(joint_radius))
    used_joints: set[int] = set()
    for a, b in MHR_BODY_EDGES:
        edge = _canonical_edge(a, b)
        if style == "unreal" and edge in _RIGHT_EDGES:
            color = right_color
        elif style == "unreal" and edge in _LEFT_EDGES:
            color = left_color
        else:
            color = default_line_color
        cv2.line(
            image,
            tuple(projected[a]),
            tuple(projected[b]),
            color,
            line_thickness,
            cv2.LINE_AA,
        )
        used_joints.update((a, b))

    if draw_joints:
        if joint_indices is None:
            joint_indices = used_joints
        else:
            joint_indices = set(int(index) for index in joint_indices)
            invalid = sorted(index for index in joint_indices if index not in used_joints)
            if invalid:
                raise ValueError(
                    f"joint_indices contains indices that are not overlay endpoints: {invalid}"
                )
        for index in sorted(joint_indices):
            cv2.circle(
                image,
                tuple(projected[index]),
                joint_radius,
                joint_color,
                -1,
                cv2.LINE_AA,
            )

    return image


__all__ = [
    "LEGACY_JOINT_COLOR",
    "LEGACY_SKELETON_COLOR",
    "POSE_OVERLAY_CHOICES",
    "UNREAL_JOINT_COLOR",
    "UNREAL_LEFT_COLOR",
    "UNREAL_RIGHT_COLOR",
    "UNREAL_SKELETON_COLOR",
    "draw_pose_overlay",
]
