"""Hip range-of-motion calculations for raw SAM-3D-Body MHR output.

For hip flexion, extension, and abduction we use the upper-leg
(hip-to-knee) vector, project it into the appropriate body plane, and measure
it from the trunk-down reference:

    reference vector: hip -> trunk-down
    measured vector:  hip -> knee, projected into the selected body plane

The positive direction is toward the face for flexion, posterior for
extension, and away from the trunk on the selected anatomical side for
abduction.  The body-aligned frame is derived from the torso, so the
calculation remains meaningful when the person leans slightly against a wall.
"""

from dataclasses import dataclass

import numpy as np

from .mhr_shoulder import MHR_BODY_EDGES, MHRBodyFrame, build_mhr_body_frame


# MHR lower-body joint indices.  ``r_upleg``/``l_upleg`` are the hip joints
# and ``r_lowleg``/``l_lowleg`` are the knees in pred_joint_coords.
MHR_HIP_JOINTS = {
    "root": 1,
    "left_hip": 2,
    "left_knee": 3,
    "left_foot": 4,
    "right_hip": 18,
    "right_knee": 19,
    "right_foot": 20,
    "c_spine3": 37,
}


# Local body_pose_params indices for the MHR upper-leg twist channels.
MHR_BODY_POSE_PARAMS = {
    "right_upleg_twist": 44,
    "left_upleg_twist": 53,
}


@dataclass(frozen=True)
class HipROMResult:
    """Geometry needed to compute and draw one hip ROM angle."""

    side: str
    measurement_name: str
    plane_name: str
    hip: np.ndarray
    knee: np.ndarray
    thigh_vector: np.ndarray
    thigh_length: float
    measurement_vector: np.ndarray
    measurement_unit: np.ndarray
    measurement_length: float
    reference_end: np.ndarray
    measurement_end: np.ndarray
    positive_axis: np.ndarray
    angle_deg: float
    body_frame: MHRBodyFrame


@dataclass(frozen=True)
class HipRotationResult:
    """Geometry for hip external/internal rotation.

    ``angle_deg`` is signed with external rotation positive.  For the
    movement named by ``rotation_name``, ``requested_angle_deg`` is positive
    in the requested direction.
    """

    side: str
    rotation_name: str
    measurement_name: str
    plane_name: str
    hip: np.ndarray
    knee: np.ndarray
    foot: np.ndarray
    thigh_vector: np.ndarray
    thigh_length: float
    shank_vector: np.ndarray
    shank_length: float
    rotation_axis: np.ndarray
    reference_vector: np.ndarray
    reference_unit: np.ndarray
    measurement_vector: np.ndarray
    measurement_unit: np.ndarray
    measurement_length: float
    reference_end: np.ndarray
    measurement_end: np.ndarray
    positive_axis: np.ndarray
    angle_deg: float
    requested_angle_deg: float
    body_frame: MHRBodyFrame


def _normalize(vector: np.ndarray, *, name: str = "vector") -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError(f"Cannot normalize near-zero {name}.")
    return vector / length


def _compute_hip_rom(
    joints: np.ndarray,
    side: str,
    plane_name: str,
    measurement_name: str,
    positive_axis: np.ndarray,
    body_frame: MHRBodyFrame | None = None,
) -> HipROMResult:
    """Compute a planar hip angle from the trunk-down reference."""

    side = side.lower().strip()
    if side not in {"right", "left"}:
        raise ValueError("side must be 'right' or 'left'")
    if plane_name not in {"sagittal", "coronal"}:
        raise ValueError("plane_name must be 'sagittal' or 'coronal'")

    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError("joints must have shape (N, 3)")
    frame = body_frame if body_frame is not None else build_mhr_body_frame(joints)
    j = MHR_HIP_JOINTS
    hip = joints[j[f"{side}_hip"]].copy()
    knee = joints[j[f"{side}_knee"]].copy()
    thigh_vector = knee - hip
    thigh_length = float(np.linalg.norm(thigh_vector))
    if thigh_length < 1e-8:
        raise ValueError(f"{side} thigh vector is too short.")

    if plane_name == "sagittal":
        # Remove abduction/adduction, leaving the component in the body
        # sagittal plane.
        measurement_vector = thigh_vector - frame.right * float(
            np.dot(thigh_vector, frame.right)
        )
    else:
        # Remove forward/backward motion, leaving the component in the body
        # coronal plane.
        measurement_vector = thigh_vector - frame.forward * float(
            np.dot(thigh_vector, frame.forward)
        )
    measurement_length = float(np.linalg.norm(measurement_vector))
    if measurement_length < 1e-8:
        raise ValueError(f"{side} thigh has no {plane_name} component.")
    measurement_unit = measurement_vector / measurement_length

    positive_axis = _normalize(positive_axis, name="hip ROM positive axis")
    angle_deg = float(
        np.degrees(
            np.arctan2(
                np.dot(measurement_unit, positive_axis),
                np.dot(measurement_unit, frame.down),
            )
        )
    )

    return HipROMResult(
        side=side,
        measurement_name=measurement_name,
        plane_name=plane_name,
        hip=hip,
        knee=knee,
        thigh_vector=thigh_vector,
        thigh_length=thigh_length,
        measurement_vector=measurement_vector,
        measurement_unit=measurement_unit,
        measurement_length=measurement_length,
        reference_end=hip + frame.down * thigh_length,
        measurement_end=hip + measurement_vector,
        positive_axis=positive_axis,
        angle_deg=angle_deg,
        body_frame=frame,
    )


def compute_hip_flexion(
    joints: np.ndarray,
    side: str = "right",
    body_frame: MHRBodyFrame | None = None,
) -> HipROMResult:
    """Compute hip flexion in the body sagittal plane."""

    frame = body_frame if body_frame is not None else build_mhr_body_frame(joints)
    return _compute_hip_rom(
        joints,
        side,
        plane_name="sagittal",
        measurement_name="hip flexion",
        positive_axis=frame.forward,
        body_frame=frame,
    )


def compute_hip_extension(
    joints: np.ndarray,
    side: str = "right",
    body_frame: MHRBodyFrame | None = None,
) -> HipROMResult:
    """Compute hip extension in the body sagittal plane.

    The displayed angle is positive toward the posterior direction.  This is
    the clinical-style magnitude for an extension trial; the same sagittal
    geometry would be negative when interpreted using the flexion-positive
    convention.
    """

    frame = body_frame if body_frame is not None else build_mhr_body_frame(joints)
    return _compute_hip_rom(
        joints,
        side,
        plane_name="sagittal",
        measurement_name="hip extension",
        positive_axis=-frame.forward,
        body_frame=frame,
    )


def compute_hip_abduction(
    joints: np.ndarray,
    side: str = "right",
    body_frame: MHRBodyFrame | None = None,
) -> HipROMResult:
    """Compute hip abduction in the body coronal plane."""

    frame = body_frame if body_frame is not None else build_mhr_body_frame(joints)
    side = side.lower().strip()
    if side not in {"right", "left"}:
        raise ValueError("side must be 'right' or 'left'")
    positive_axis = frame.right if side == "right" else -frame.right
    return _compute_hip_rom(
        joints,
        side,
        plane_name="coronal",
        measurement_name="hip abduction",
        positive_axis=positive_axis,
        body_frame=frame,
    )


def select_hip_side(
    joints: np.ndarray,
    side: str = "auto",
    body_frame: MHRBodyFrame | None = None,
) -> str:
    """Return the requested side, or the side with the larger flexion angle."""

    side = side.lower().strip()
    if side in {"right", "left"}:
        return side
    if side != "auto":
        raise ValueError("side must be 'auto', 'right', or 'left'")

    frame = body_frame if body_frame is not None else build_mhr_body_frame(joints)
    candidates = [
        compute_hip_flexion(joints, "right", frame),
        compute_hip_flexion(joints, "left", frame),
    ]
    return max(candidates, key=lambda result: abs(result.angle_deg)).side


def _project_perpendicular(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Remove the component of ``vector`` parallel to ``normal``."""

    normal = _normalize(normal, name="projection normal")
    vector = np.asarray(vector, dtype=np.float64)
    return vector - normal * float(np.dot(vector, normal))


def compute_hip_rotation(
    joints: np.ndarray,
    side: str = "right",
    rotation: str = "external",
    body_frame: MHRBodyFrame | None = None,
) -> HipRotationResult:
    """Compute hip axial rotation from the bent lower leg.

    This is intended for the standing leg-raise setup in which the hip is
    flexed and the knee is approximately 90 degrees.  The thigh is the
    rotation axis; the knee-to-foot (shank) vector is projected into the plane
    perpendicular to that axis.  A body-down reference represents the neutral
    lower-leg direction when the thigh is held forward.  The subject's
    anatomical outward direction defines positive external rotation.
    """

    side = side.lower().strip()
    if side not in {"right", "left"}:
        raise ValueError("side must be 'right' or 'left'")
    rotation = rotation.lower().strip().replace("_rotation", "")
    if rotation not in {"external", "internal"}:
        raise ValueError("rotation must be 'external' or 'internal'")

    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError("joints must have shape (N, 3)")
    frame = body_frame if body_frame is not None else build_mhr_body_frame(joints)
    j = MHR_HIP_JOINTS
    hip = joints[j[f"{side}_hip"]].copy()
    knee = joints[j[f"{side}_knee"]].copy()
    foot = joints[j[f"{side}_foot"]].copy()

    thigh_vector = knee - hip
    thigh_length = float(np.linalg.norm(thigh_vector))
    if thigh_length < 1e-8:
        raise ValueError(f"{side} thigh vector is too short.")
    thigh_unit = thigh_vector / thigh_length

    shank_vector = foot - knee
    shank_length = float(np.linalg.norm(shank_vector))
    if shank_length < 1e-8:
        raise ValueError(f"{side} shank vector is too short.")

    measurement_vector = _project_perpendicular(shank_vector, thigh_unit)
    measurement_length = float(np.linalg.norm(measurement_vector))
    if measurement_length < 1e-8:
        raise ValueError(f"{side} shank has no transverse component.")
    measurement_unit = measurement_vector / measurement_length

    # With the thigh held forward, a downward shank is the neutral reference.
    reference_vector = _project_perpendicular(frame.down, thigh_unit)
    reference_unit = _normalize(reference_vector, name="hip-rotation reference")

    # Orthogonalize anatomical outward against the neutral reference.  This
    # gives a clearly interpretable positive direction even when the subject
    # leans or the thigh is not perfectly horizontal.
    outward = frame.right if side == "right" else -frame.right
    positive_axis = _project_perpendicular(outward, thigh_unit)
    positive_axis = positive_axis - reference_unit * float(np.dot(positive_axis, reference_unit))
    positive_axis = _normalize(positive_axis, name="external hip-rotation direction")
    rotation_axis = _normalize(
        np.cross(reference_unit, positive_axis), name="hip rotation axis"
    )

    angle_deg = float(
        np.degrees(
            np.arctan2(
                np.dot(positive_axis, measurement_unit),
                np.dot(reference_unit, measurement_unit),
            )
        )
    )
    requested_angle_deg = angle_deg if rotation == "external" else -angle_deg

    return HipRotationResult(
        side=side,
        rotation_name=rotation,
        measurement_name=f"hip {rotation} rotation",
        plane_name="thigh transverse",
        hip=hip,
        knee=knee,
        foot=foot,
        thigh_vector=thigh_vector,
        thigh_length=thigh_length,
        shank_vector=shank_vector,
        shank_length=shank_length,
        rotation_axis=rotation_axis,
        reference_vector=reference_unit * measurement_length,
        reference_unit=reference_unit,
        measurement_vector=measurement_vector,
        measurement_unit=measurement_unit,
        measurement_length=measurement_length,
        reference_end=knee + reference_unit * measurement_length,
        measurement_end=knee + measurement_vector,
        positive_axis=positive_axis,
        angle_deg=angle_deg,
        requested_angle_deg=requested_angle_deg,
        body_frame=frame,
    )


def mhr_upleg_twist_index(side: str) -> int:
    """Return the body_pose_params index for the selected upper-leg twist."""

    side = side.lower().strip()
    if side not in {"right", "left"}:
        raise ValueError("side must be 'right' or 'left'")
    return MHR_BODY_POSE_PARAMS[f"{side}_upleg_twist"]


def mhr_upleg_twist_deg(body_pose_params: np.ndarray, side: str) -> float:
    """Read the MHR local upper-leg twist channel in degrees."""

    params = np.asarray(body_pose_params, dtype=np.float64).reshape(-1)
    index = mhr_upleg_twist_index(side)
    if index >= len(params):
        raise ValueError(f"body_pose_params has no index {index}: shape {params.shape}")
    return float(np.degrees(params[index]))


__all__ = [
    "MHR_BODY_EDGES",
    "MHR_BODY_POSE_PARAMS",
    "MHR_HIP_JOINTS",
    "HipROMResult",
    "HipRotationResult",
    "compute_hip_abduction",
    "compute_hip_extension",
    "compute_hip_flexion",
    "compute_hip_rotation",
    "mhr_upleg_twist_deg",
    "mhr_upleg_twist_index",
    "select_hip_side",
]
