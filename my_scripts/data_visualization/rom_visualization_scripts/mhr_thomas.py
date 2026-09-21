"""Body-relative measurements for the modified Thomas test.

The clinical table is not available in a monocular SAM-3D-Body output.  When
the subject is assumed to be supine and flat, the plane spanned by the trunk
axis and the subject's left/right axis is used as an estimate of the table
plane.  The normal of that plane is the body-forward direction.

The primary hip value is a sagittal, body-relative extension proxy:

    0 degrees  = the thigh follows the body-down direction
    positive   = the thigh is below the inferred table plane (extension)
    negative   = the thigh is above the inferred table plane (flexion)

The separate ``thigh_elevation_deg`` value reports the signed angle from the
inferred table plane, and ``thigh_abduction_deg`` reports lateral drift.  The
measurements are geometric estimates, not direct observations of table
contact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mhr_hip import MHR_HIP_JOINTS
from .mhr_shoulder import MHRBodyFrame, MHR_JOINTS, build_mhr_body_frame


MHR_THOMAS_JOINTS = {
    "root": MHR_HIP_JOINTS["root"],
    "c_spine3": MHR_HIP_JOINTS["c_spine3"],
    "left_hip": MHR_HIP_JOINTS["left_hip"],
    "left_knee": MHR_HIP_JOINTS["left_knee"],
    "left_ankle": MHR_HIP_JOINTS["left_foot"],
    "right_hip": MHR_HIP_JOINTS["right_hip"],
    "right_knee": MHR_HIP_JOINTS["right_knee"],
    "right_ankle": MHR_HIP_JOINTS["right_foot"],
}


def build_thomas_body_frame(joints: np.ndarray) -> MHRBodyFrame:
    """Estimate a supine body/table frame from the trunk and pelvis.

    ``build_mhr_body_frame`` uses one upper-trunk vector and only the shoulder
    axis.  That is sensitive when the patient curls the shoulders forward to
    hold the opposite knee.  For this test, use a least-squares plane fit over
    the central spine, pelvis, and shoulder landmarks, then derive the body
    axes inside that plane.  The multi-landmark fit is more stable than
    selecting either the shoulders or the lower back alone, but it is still a
    body-only estimate rather than a detected table surface.
    """

    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 3 or joints.shape[0] <= 114:
        raise ValueError("joints must have shape (127, 3) or another (N, 3) with N > 114")

    j = MHR_JOINTS
    plane_indices = [
        j["root"],
        j["c_spine0"],
        j["c_spine1"],
        j["c_spine2"],
        j["c_spine3"],
        MHR_THOMAS_JOINTS["left_hip"],
        MHR_THOMAS_JOINTS["right_hip"],
        j["r_shoulder"],
        j["l_shoulder"],
    ]
    plane_points = joints[plane_indices]
    _, _, vh = np.linalg.svd(plane_points - plane_points.mean(axis=0), full_matrices=False)
    plane_normal = _normalize(vh[-1], name="torso/pelvis plane normal")

    headward = joints[j["c_spine3"]] - joints[j["root"]]
    up = _normalize(
        headward - plane_normal * float(np.dot(headward, plane_normal)),
        name="torso axis in body plane",
    )

    pelvis_axis = joints[MHR_THOMAS_JOINTS["right_hip"]] - joints[MHR_THOMAS_JOINTS["left_hip"]]
    shoulder_axis = joints[j["r_shoulder"]] - joints[j["l_shoulder"]]
    right_raw = 0.5 * (pelvis_axis + shoulder_axis)
    right = right_raw - plane_normal * float(np.dot(right_raw, plane_normal))
    right = right - up * float(np.dot(right, up))
    right = _normalize(right, name="body right axis in body plane")
    if float(np.dot(right, right_raw)) < 0.0:
        right = -right

    forward = _normalize(np.cross(right, up), name="body plane normal")
    face_direction = joints[j["c_jaw"]] - joints[j["c_head"]]
    if float(np.linalg.norm(face_direction)) >= 1e-8:
        face_direction = _normalize(face_direction, name="face direction")
        if float(np.dot(forward, face_direction)) < 0.0:
            forward = -forward

    return MHRBodyFrame(
        up=up,
        down=-up,
        right=right,
        forward=forward,
    )


@dataclass(frozen=True)
class ThomasResult:
    """Geometry and measurements for one candidate test leg."""

    side: str
    hip: np.ndarray
    knee: np.ndarray
    ankle: np.ndarray
    thigh_vector: np.ndarray
    shank_vector: np.ndarray
    thigh_length: float
    shank_length: float
    sagittal_vector: np.ndarray
    coronal_vector: np.ndarray
    sagittal_unit: np.ndarray
    coronal_unit: np.ndarray
    reference_unit: np.ndarray
    extension_axis: np.ndarray
    abduction_axis: np.ndarray
    hip_extension_deg: float
    thigh_elevation_deg: float
    knee_flexion_deg: float
    thigh_abduction_deg: float
    body_frame: MHRBodyFrame

    @property
    def knee_deviation_from_90_deg(self) -> float:
        return self.knee_flexion_deg - 90.0

    @property
    def reference_end(self) -> np.ndarray:
        return self.hip + self.reference_unit * self.thigh_length

    @property
    def sagittal_end(self) -> np.ndarray:
        return self.hip + self.sagittal_unit * self.thigh_length

    @property
    def coronal_end(self) -> np.ndarray:
        return self.hip + self.coronal_unit * self.thigh_length


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError(f"Cannot normalize near-zero {name}.")
    return vector / length


def _project_onto_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    normal = _normalize(normal, name="plane normal")
    vector = np.asarray(vector, dtype=np.float64)
    return vector - normal * float(np.dot(vector, normal))


def _signed_angle_from_reference(
    vector: np.ndarray,
    reference: np.ndarray,
    positive_axis: np.ndarray,
) -> float:
    """Return an angle in the plane spanned by reference and positive_axis."""

    vector = _normalize(vector, name="measurement vector")
    reference = _normalize(reference, name="reference vector")
    positive_axis = _normalize(positive_axis, name="positive axis")
    return float(
        np.degrees(
            np.arctan2(
                float(np.dot(vector, positive_axis)),
                float(np.dot(vector, reference)),
            )
        )
    )


def _angle_to_plane(vector: np.ndarray, normal: np.ndarray) -> float:
    """Return signed elevation relative to a plane, in degrees."""

    vector = _normalize(vector, name="vector")
    normal = _normalize(normal, name="plane normal")
    in_plane = vector - normal * float(np.dot(vector, normal))
    return float(
        np.degrees(
            np.arctan2(float(np.dot(vector, normal)), float(np.linalg.norm(in_plane)))
        )
    )


def compute_thomas_result(
    joints: np.ndarray,
    side: str = "right",
    body_frame: MHRBodyFrame | None = None,
) -> ThomasResult:
    """Compute the body-relative modified Thomas measurements."""

    side = side.lower().strip()
    if side not in {"right", "left"}:
        raise ValueError("side must be 'right' or 'left'")
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError("joints must have shape (N, 3)")

    frame = body_frame if body_frame is not None else build_mhr_body_frame(joints)
    prefix = side[0]
    hip = joints[MHR_THOMAS_JOINTS[f"{side}_hip"]].copy()
    knee = joints[MHR_THOMAS_JOINTS[f"{side}_knee"]].copy()
    ankle = joints[MHR_THOMAS_JOINTS[f"{side}_ankle"]].copy()
    thigh_vector = knee - hip
    shank_vector = ankle - knee
    thigh_length = float(np.linalg.norm(thigh_vector))
    shank_length = float(np.linalg.norm(shank_vector))
    if thigh_length < 1e-8:
        raise ValueError(f"{side} thigh vector is too short.")
    if shank_length < 1e-8:
        raise ValueError(f"{side} shank vector is too short.")

    # The inferred table/body plane is spanned by body-down and body-right.
    # The sagittal plane is spanned by body-down and body-forward; the coronal
    # plane is spanned by body-down and body-right.
    reference_unit = frame.down
    extension_axis = -frame.forward  # posterior/below-table is positive
    outward = frame.right if prefix == "r" else -frame.right

    sagittal_vector = _project_onto_plane(thigh_vector, frame.right)
    coronal_vector = _project_onto_plane(thigh_vector, frame.forward)
    sagittal_unit = _normalize(sagittal_vector, name=f"{side} sagittal thigh")
    coronal_unit = _normalize(coronal_vector, name=f"{side} coronal thigh")

    hip_extension_deg = _signed_angle_from_reference(
        sagittal_unit,
        reference_unit,
        extension_axis,
    )
    thigh_elevation_deg = _angle_to_plane(thigh_vector, frame.forward)
    knee_flexion_deg = float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(_normalize(hip - knee, name="thigh-at-knee"),
                           _normalize(ankle - knee, name="shank-at-knee")),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    thigh_abduction_deg = _signed_angle_from_reference(
        coronal_unit,
        reference_unit,
        outward,
    )

    return ThomasResult(
        side=side,
        hip=hip,
        knee=knee,
        ankle=ankle,
        thigh_vector=thigh_vector,
        shank_vector=shank_vector,
        thigh_length=thigh_length,
        shank_length=shank_length,
        sagittal_vector=sagittal_vector,
        coronal_vector=coronal_vector,
        sagittal_unit=sagittal_unit,
        coronal_unit=coronal_unit,
        reference_unit=reference_unit,
        extension_axis=extension_axis,
        abduction_axis=outward,
        hip_extension_deg=hip_extension_deg,
        thigh_elevation_deg=thigh_elevation_deg,
        knee_flexion_deg=knee_flexion_deg,
        thigh_abduction_deg=thigh_abduction_deg,
        body_frame=frame,
    )


def select_thomas_side(
    joints: np.ndarray,
    side: str = "auto",
    body_frame: MHRBodyFrame | None = None,
) -> str:
    """Select the leg most consistent with the hanging/test leg.

    In this setup the held-to-chest leg has a larger positive body-forward
    component, while the hanging leg has the lower sagittal elevation.  This
    is only a convenience heuristic; a supplied anatomical side is preferred
    when it is known.
    """

    side = side.lower().strip()
    if side in {"right", "left"}:
        return side
    if side != "auto":
        raise ValueError("side must be 'auto', 'right', or 'left'")

    frame = body_frame if body_frame is not None else build_mhr_body_frame(joints)
    candidates = [
        compute_thomas_result(joints, "right", frame),
        compute_thomas_result(joints, "left", frame),
    ]
    return min(candidates, key=lambda result: result.thigh_elevation_deg).side


__all__ = [
    "MHR_THOMAS_JOINTS",
    "ThomasResult",
    "build_thomas_body_frame",
    "compute_thomas_result",
    "select_thomas_side",
]
