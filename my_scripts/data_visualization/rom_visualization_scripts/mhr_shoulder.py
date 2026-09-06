"""Shoulder range-of-motion calculations for raw SAM-3D-Body MHR output.

The raw SAM-3D-Body file stores 127 MHR joint positions in camera coordinates.
For shoulder flexion we use the MHR upper-arm joint to elbow joint and measure
it in the subject's sagittal plane:

    reference vector: shoulder -> trunk-down
    measured vector:  shoulder -> elbow, projected onto the sagittal plane

The signed angle is positive toward the subject's face (forward flexion) and
negative behind the subject (extension).  This is deliberately kept separate
from the drawing code so the same calculation can be used for plots or batch
ROM extraction later.

For shoulder axial rotation, the elbow is assumed to be flexed.  We project
the forearm direction into the plane perpendicular to the humerus and measure
it from the body-forward direction.  The sign convention is external rotation
positive and internal rotation negative.  This is a useful image-based ROM
estimate; the MHR ``r/l_uparm_twist`` channel is also available, but it is a
model-local twist parameter rather than a clinical goniometer angle.
"""

from dataclasses import dataclass

import numpy as np


# Joint indices in the 127-joint MHR skeleton.  These are the indices used by
# pred_joint_coords in the SAM-3D-Body MHR NPZ files.
MHR_JOINTS = {
    "root": 1,
    "c_spine0": 34,
    "c_spine1": 35,
    "c_spine2": 36,
    "c_spine3": 37,
    "r_clavicle": 38,
    "r_shoulder": 39,  # r_uparm
    "r_elbow": 40,     # r_lowarm
    "r_wrist": 42,
    "l_clavicle": 74,
    "l_shoulder": 75,  # l_uparm
    "l_elbow": 76,      # l_lowarm
    "l_wrist": 78,
    "c_neck": 110,
    "c_head": 113,
    "c_jaw": 114,
}


# Indices are local to body_pose_params (the six global rigid parameters are
# not included in that array).  These are the MHR channels most directly
# associated with humeral axial twist.
MHR_BODY_POSE_PARAMS = {
    "right_uparm_twist": 27,
    "left_uparm_twist": 37,
}


# A compact body skeleton for the mesh panels.  The intermediate hand and foot
# joints are intentionally omitted because they obscure the ROM annotation.
MHR_BODY_EDGES = (
    (1, 34), (34, 35), (35, 36), (36, 37),
    (1, 2), (2, 3), (3, 4), (4, 8),
    (1, 18), (18, 19), (19, 20), (20, 24),
    (37, 38), (38, 39), (39, 40), (40, 42),
    (37, 74), (74, 75), (75, 76), (76, 78),
    (37, 110), (110, 113), (113, 114),
)


def _normalize(vector: np.ndarray, *, name: str = "vector") -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError(f"Cannot normalize near-zero {name}.")
    return vector / length


@dataclass(frozen=True)
class MHRBodyFrame:
    """Subject-centered orthonormal directions in stored MHR coordinates."""

    up: np.ndarray
    down: np.ndarray
    right: np.ndarray
    forward: np.ndarray


@dataclass(frozen=True)
class ShoulderROMResult:
    """All geometry needed to compute and draw one shoulder ROM angle."""

    side: str
    measurement_name: str
    plane_name: str
    shoulder: np.ndarray
    elbow: np.ndarray
    arm_vector: np.ndarray
    arm_length: float
    measurement_vector: np.ndarray
    measurement_unit: np.ndarray
    measurement_length: float
    reference_end: np.ndarray
    measurement_end: np.ndarray
    positive_axis: np.ndarray
    angle_deg: float
    body_frame: MHRBodyFrame


@dataclass(frozen=True)
class ShoulderFlexionResult(ShoulderROMResult):
    """Geometry for shoulder flexion in the body sagittal plane."""

    @property
    def sagittal_vector(self) -> np.ndarray:
        return self.measurement_vector

    @property
    def sagittal_unit(self) -> np.ndarray:
        return self.measurement_unit

    @property
    def sagittal_length(self) -> float:
        return self.measurement_length

    @property
    def sagittal_end(self) -> np.ndarray:
        return self.measurement_end


@dataclass(frozen=True)
class ShoulderAbductionResult(ShoulderROMResult):
    """Geometry for shoulder abduction in the body coronal plane."""

    @property
    def coronal_vector(self) -> np.ndarray:
        return self.measurement_vector

    @property
    def coronal_unit(self) -> np.ndarray:
        return self.measurement_unit

    @property
    def coronal_length(self) -> float:
        return self.measurement_length

    @property
    def coronal_end(self) -> np.ndarray:
        return self.measurement_end


@dataclass(frozen=True)
class ShoulderRotationResult:
    """Geometry for shoulder external/internal rotation.

    ``angle_deg`` is signed with external rotation positive.  For the
    movement named by ``rotation_name``, ``requested_angle_deg`` is positive
    in the requested direction, which is convenient for a clinical-style
    readout (external and internal ROM are usually reported separately).
    """

    side: str
    rotation_name: str
    measurement_name: str
    plane_name: str
    shoulder: np.ndarray
    elbow: np.ndarray
    wrist: np.ndarray
    arm_vector: np.ndarray
    arm_length: float
    forearm_vector: np.ndarray
    forearm_length: float
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


def build_mhr_body_frame(joints: np.ndarray) -> MHRBodyFrame:
    """Build up/right/forward directions from one MHR joint array.

    ``pred_joint_coords`` has the MHR camera-system y/z sign flip applied.  We
    do not hard-code that flip here; instead, the trunk and shoulder geometry
    defines up/right, while the head-to-jaw vector selects the forward sign.
    This keeps the function usable for poses with a rotated torso.
    """

    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 3 or joints.shape[0] <= 114:
        raise ValueError("joints must have shape (127, 3) or another (N, 3) with N > 114")

    j = MHR_JOINTS
    up = _normalize(joints[j["c_spine3"]] - joints[j["root"]], name="trunk axis")

    # MHR's r_* joints are on the subject's anatomical right.  Orthogonalize
    # the shoulder-to-shoulder direction against the trunk axis so the
    # sagittal plane is stable when the subject leans.
    right_raw = joints[j["r_shoulder"]] - joints[j["l_shoulder"]]
    right = right_raw - up * float(np.dot(right_raw, up))
    right = _normalize(right, name="shoulder axis")

    forward = _normalize(np.cross(right, up), name="forward axis")
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


def _compute_shoulder_rom(
    joints: np.ndarray,
    side: str,
    plane_name: str,
    measurement_name: str,
    result_type,
    body_frame: MHRBodyFrame | None = None,
) -> ShoulderROMResult:
    side = side.lower().strip()
    if side not in {"right", "left"}:
        raise ValueError("side must be 'right' or 'left'")
    if plane_name not in {"sagittal", "coronal"}:
        raise ValueError("plane_name must be 'sagittal' or 'coronal'")

    joints = np.asarray(joints, dtype=np.float64)
    frame = body_frame if body_frame is not None else build_mhr_body_frame(joints)
    j = MHR_JOINTS
    shoulder = joints[j[f"{side[0]}_shoulder"]].copy()
    elbow = joints[j[f"{side[0]}_elbow"]].copy()
    arm_vector = elbow - shoulder
    arm_length = float(np.linalg.norm(arm_vector))
    if arm_length < 1e-8:
        raise ValueError(f"{side} upper-arm vector is too short.")

    if plane_name == "sagittal":
        # Remove the lateral/abduction component.  Positive is toward the
        # face, i.e. forward flexion.
        measurement_vector = arm_vector - frame.right * float(np.dot(arm_vector, frame.right))
        positive_axis = frame.forward
    else:
        # Remove the forward/backward component.  Positive is away from the
        # trunk on the selected anatomical side, i.e. abduction.
        measurement_vector = arm_vector - frame.forward * float(np.dot(arm_vector, frame.forward))
        positive_axis = frame.right if side == "right" else -frame.right

    measurement_length = float(np.linalg.norm(measurement_vector))
    if measurement_length < 1e-8:
        raise ValueError(f"{side} upper-arm vector has no {plane_name} component.")
    measurement_unit = measurement_vector / measurement_length

    # Zero is arm-down.  atan2 preserves the sign toward the movement's
    # positive axis, unlike an unsigned arccos angle.
    positive_component = float(np.dot(measurement_unit, positive_axis))
    down_component = float(np.dot(measurement_unit, frame.down))
    angle_deg = float(np.degrees(np.arctan2(positive_component, down_component)))

    return result_type(
        side=side,
        measurement_name=measurement_name,
        plane_name=plane_name,
        shoulder=shoulder,
        elbow=elbow,
        arm_vector=arm_vector,
        arm_length=arm_length,
        measurement_vector=measurement_vector,
        measurement_unit=measurement_unit,
        measurement_length=measurement_length,
        reference_end=shoulder + frame.down * arm_length,
        measurement_end=shoulder + measurement_vector,
        positive_axis=positive_axis,
        angle_deg=angle_deg,
        body_frame=frame,
    )


def compute_shoulder_flexion(
    joints: np.ndarray,
    side: str,
    body_frame: MHRBodyFrame | None = None,
) -> ShoulderFlexionResult:
    """Compute signed shoulder flexion in the body sagittal plane."""

    return _compute_shoulder_rom(
        joints,
        side,
        plane_name="sagittal",
        measurement_name="shoulder flexion",
        result_type=ShoulderFlexionResult,
        body_frame=body_frame,
    )


def compute_shoulder_abduction(
    joints: np.ndarray,
    side: str,
    body_frame: MHRBodyFrame | None = None,
) -> ShoulderAbductionResult:
    """Compute signed shoulder abduction in the body coronal plane."""

    return _compute_shoulder_rom(
        joints,
        side,
        plane_name="coronal",
        measurement_name="shoulder abduction",
        result_type=ShoulderAbductionResult,
        body_frame=body_frame,
    )


def _project_perpendicular(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Remove the component of ``vector`` parallel to unit ``normal``."""

    vector = np.asarray(vector, dtype=np.float64)
    normal = _normalize(normal, name="projection normal")
    return vector - normal * float(np.dot(vector, normal))


def compute_shoulder_rotation(
    joints: np.ndarray,
    side: str,
    rotation: str = "external",
    body_frame: MHRBodyFrame | None = None,
) -> ShoulderRotationResult:
    """Compute signed shoulder axial rotation from the bent forearm.

    The reference is body-forward projected into the plane perpendicular to
    the humerus.  ``rotation_axis`` is oriented so that moving from the
    reference toward the subject's anatomical outward direction is positive
    for both arms.  This makes external rotation positive and internal
    rotation negative regardless of side.

    The method is best suited to the standard rotation test setup: the elbow
    is flexed approximately 90 degrees and kept close to the torso.  It is
    intentionally based on shoulder/elbow/wrist geometry instead of treating
    the MHR local ``uparm_twist`` value as a clinical angle.
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
    j = MHR_JOINTS
    shoulder = joints[j[f"{side[0]}_shoulder"]].copy()
    elbow = joints[j[f"{side[0]}_elbow"]].copy()
    wrist = joints[j[f"{side[0]}_wrist"]].copy()

    arm_vector = elbow - shoulder
    arm_length = float(np.linalg.norm(arm_vector))
    if arm_length < 1e-8:
        raise ValueError(f"{side} upper-arm vector is too short.")
    arm_unit = arm_vector / arm_length

    forearm_vector = wrist - elbow
    forearm_length = float(np.linalg.norm(forearm_vector))
    if forearm_length < 1e-8:
        raise ValueError(f"{side} forearm vector is too short.")

    # Remove elbow-flexion direction along the humerus.  In the intended
    # approximately-90-degree setup this changes the vector very little, but
    # it makes the axial angle stable when the elbow is not exactly 90 deg.
    measurement_vector = _project_perpendicular(forearm_vector, arm_unit)
    measurement_length = float(np.linalg.norm(measurement_vector))
    if measurement_length < 1e-8:
        raise ValueError(f"{side} forearm has no transverse component.")
    measurement_unit = measurement_vector / measurement_length

    reference_vector = _project_perpendicular(frame.forward, arm_unit)
    if np.linalg.norm(reference_vector) < 1e-8:
        # A humerus pointing directly forward is degenerate for a forward
        # reference; anatomical outward is the least surprising fallback.
        outward = frame.right if side == "right" else -frame.right
        reference_vector = _project_perpendicular(outward, arm_unit)
    reference_unit = _normalize(reference_vector, name="rotation reference")

    # Reverse the left humerus axis so an outward forearm sweep is positive on
    # both sides.  cross(axis, reference) is the positive transverse direction.
    rotation_axis = arm_unit if side == "right" else -arm_unit
    positive_axis = _normalize(
        np.cross(rotation_axis, reference_unit), name="external rotation axis"
    )
    angle_deg = float(
        np.degrees(
            np.arctan2(
                np.dot(rotation_axis, np.cross(reference_unit, measurement_unit)),
                np.dot(reference_unit, measurement_unit),
            )
        )
    )
    requested_angle_deg = angle_deg if rotation == "external" else -angle_deg

    measurement_name = f"shoulder {rotation} rotation"
    return ShoulderRotationResult(
        side=side,
        rotation_name=rotation,
        measurement_name=measurement_name,
        plane_name="humerus transverse",
        shoulder=shoulder,
        elbow=elbow,
        wrist=wrist,
        arm_vector=arm_vector,
        arm_length=arm_length,
        forearm_vector=forearm_vector,
        forearm_length=forearm_length,
        rotation_axis=rotation_axis,
        reference_vector=reference_unit * measurement_length,
        reference_unit=reference_unit,
        measurement_vector=measurement_vector,
        measurement_unit=measurement_unit,
        measurement_length=measurement_length,
        reference_end=elbow + reference_unit * measurement_length,
        measurement_end=elbow + measurement_vector,
        positive_axis=positive_axis,
        angle_deg=angle_deg,
        requested_angle_deg=requested_angle_deg,
        body_frame=frame,
    )


def mhr_uparm_twist_index(side: str) -> int:
    """Return the body_pose_params index for the selected upper-arm twist."""

    side = side.lower().strip()
    if side not in {"right", "left"}:
        raise ValueError("side must be 'right' or 'left'")
    return MHR_BODY_POSE_PARAMS[f"{side}_uparm_twist"]


def mhr_uparm_twist_deg(body_pose_params: np.ndarray, side: str) -> float:
    """Read the MHR local upper-arm twist channel in degrees."""

    params = np.asarray(body_pose_params, dtype=np.float64).reshape(-1)
    index = mhr_uparm_twist_index(side)
    if index >= len(params):
        raise ValueError(f"body_pose_params has no index {index}: shape {params.shape}")
    return float(np.degrees(params[index]))


def select_shoulder_side(
    joints: np.ndarray,
    side: str = "auto",
    movement: str = "flexion",
) -> str:
    """Return the requested side, or the side with the largest motion."""

    side = side.lower().strip()
    if side in {"right", "left"}:
        return side
    if side != "auto":
        raise ValueError("side must be 'auto', 'right', or 'left'")

    frame = build_mhr_body_frame(joints)
    movement = movement.lower().strip()
    if movement in {"external", "external_rotation", "internal", "internal_rotation"}:
        rotation = "external" if movement.startswith("external") else "internal"
        candidates = [
            compute_shoulder_rotation(joints, "right", rotation, frame),
            compute_shoulder_rotation(joints, "left", rotation, frame),
        ]
        # The standard shoulder-rotation setup has the elbow near 90 degrees.
        # Prefer that arm so a relaxed contralateral arm is not selected just
        # because its forearm happens to point toward the body-forward axis.
        scored = []
        for candidate in candidates:
            upper = _normalize(candidate.shoulder - candidate.elbow, name="upper-arm")
            forearm = _normalize(candidate.wrist - candidate.elbow, name="forearm")
            elbow_angle = float(np.degrees(np.arccos(np.clip(np.dot(upper, forearm), -1.0, 1.0))))
            scored.append((-abs(elbow_angle - 90.0), abs(candidate.angle_deg), candidate))
        return max(scored, key=lambda item: (item[0], item[1]))[2].side

    compute = compute_shoulder_abduction if movement == "abduction" else compute_shoulder_flexion
    candidates = [
        compute(joints, "right", frame),
        compute(joints, "left", frame),
    ]
    return max(candidates, key=lambda result: abs(result.angle_deg)).side
