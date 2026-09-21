"""Pure-Python port of the supplied Unreal ``M_SkinFresnel`` material.

The Unreal material is translucent. Its active graph computes opacity from the
camera vector and vertex normal, then uses the blue-gray ``EmissiveCol``
material-instance value for the visible colour::

    opacity = pow(
        1 - saturate(abs(dot(CameraVectorWS, VertexNormalWS))),
        FresnelExponent,
    ) * FresnelStrength

The values below were read from ``M_SkinFresnel.uasset`` and
``MI_SkinFresnel.uasset``. Unreal assets are not loaded at runtime: the graph
is evaluated on the CPU and passed to ``pyrender`` as interpolated vertex
colour/alpha data. MHR vertices and topology are never modified.
"""

from __future__ import annotations

import numpy as np
import trimesh


SKIN_CHOICES = ("original", "reskinned")

# Keep the original renderer's appearance as the compatibility/default mode.
ORIGINAL_MESH_COLOR = np.asarray([174, 197, 226, 255], dtype=np.uint8)

# ``MI_SkinFresnel`` overrides the master material's EmissiveCol parameter.
# These are linear Unreal LinearColor values, not 0-255 sRGB values.
RESKIN_EMISSIVE_COLOR = np.asarray(
    [0.6467739939689636, 0.7424389719963074, 0.8385419845581055],
    dtype=np.float64,
)

# The active opacity chain in M_SkinFresnel uses these master defaults.
RESKIN_FRESNEL_EXPONENT = 5.0
RESKIN_FRESNEL_STRENGTH = 1.0


def make_render_material(skin: str = "original"):
    """Return the ``pyrender`` material corresponding to ``skin``.

    The reskinned Unreal material has a black base-color path, an emissive
    output, and translucent alpha. ``pyrender`` has no Unreal material
    equivalent, so a metallic-roughness material with black base color and a
    unit emissive factor is used; per-vertex RGB carries the Unreal
    EmissiveCol value and per-vertex alpha carries the graph output.
    ``None`` preserves the original renderer's material behavior.
    """

    if skin not in SKIN_CHOICES:
        choices = ", ".join(SKIN_CHOICES)
        raise ValueError(f"Unknown MHR skin {skin!r}; choose {choices}.")
    if skin == "original":
        return None

    # Import lazily so geometry-only tests do not initialize OpenGL.
    import pyrender

    return pyrender.MetallicRoughnessMaterial(
        name="M_SkinFresnel_python",
        alphaMode="BLEND",
        baseColorFactor=[0.0, 0.0, 0.0, 1.0],
        emissiveFactor=[1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
        smooth=True,
    )


def make_vertex_colors(
    vertices: np.ndarray,
    faces: np.ndarray,
    skin: str = "original",
    view_axis: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> np.ndarray:
    """Return RGBA colours without changing mesh geometry.

    ``vertices`` are expected in the render-camera coordinate system used by
    the existing ``pyrender`` renderer.  The camera looks approximately along
    +Z because the orthographic camera is placed at positive Z.

    Original mode returns uint8 colours for compatibility. Reskinned mode
    returns float32 colours in normalized ``[0, 1]`` form so the material
    instance colour and translucent alpha are not unnecessarily quantized.
    """

    if skin not in SKIN_CHOICES:
        choices = ", ".join(SKIN_CHOICES)
        raise ValueError(f"Unknown MHR skin {skin!r}; choose {choices}.")

    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if skin == "original":
        return np.tile(ORIGINAL_MESH_COLOR[None, :], (len(vertices), 1))

    if view_axis is None:
        view_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    view_axis = np.asarray(view_axis, dtype=np.float64).reshape(3)
    view_norm = float(np.linalg.norm(view_axis))
    if not np.isfinite(view_norm) or view_norm < 1e-8:
        raise ValueError(f"view_axis must be a finite non-zero vector, got {view_axis!r}.")
    view_axis /= view_norm

    # Compute smooth per-vertex normals with the same topology that will be
    # rendered.  This is a small CPU-side preparation step; the geometry is
    # otherwise exactly the MHR geometry returned by the model.
    if normals is None:
        normal_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        normals = normal_mesh.vertex_normals
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape != vertices.shape:
        raise ValueError(
            f"normals must have shape {vertices.shape}, got {normals.shape}."
        )
    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_norms, 1e-12)

    # This is the active Unreal material graph, evaluated at vertices so
    # OpenGL interpolates it across each triangle. The absolute value matches
    # the graph and keeps the two-sided render visually symmetric.
    camera_normal_dot = np.clip(np.abs(normals @ view_axis), 0.0, 1.0)
    opacity = np.power(1.0 - camera_normal_dot, RESKIN_FRESNEL_EXPONENT)
    opacity = np.clip(RESKIN_FRESNEL_STRENGTH * opacity, 0.0, 1.0)

    rgb = np.tile(RESKIN_EMISSIVE_COLOR[None, :], (len(vertices), 1))
    return np.column_stack((rgb, opacity)).astype(np.float32)


def compare_same_geometry(
    original_vertices: np.ndarray,
    reskinned_vertices: np.ndarray,
    original_faces: np.ndarray,
    reskinned_faces: np.ndarray,
) -> dict[str, float | int | bool]:
    """Validate the appearance-only contract of the two render modes."""

    original_vertices = np.asarray(original_vertices, dtype=np.float64)
    reskinned_vertices = np.asarray(reskinned_vertices, dtype=np.float64)
    original_faces = np.asarray(original_faces, dtype=np.int64)
    reskinned_faces = np.asarray(reskinned_faces, dtype=np.int64)

    if original_vertices.shape != reskinned_vertices.shape:
        max_abs = float("inf")
        mean_abs = float("inf")
    else:
        delta = np.abs(original_vertices - reskinned_vertices)
        max_abs = float(delta.max(initial=0.0))
        mean_abs = float(delta.mean())

    return {
        "vertex_count_original": int(len(original_vertices)),
        "vertex_count_reskinned": int(len(reskinned_vertices)),
        "face_count_original": int(len(original_faces)),
        "face_count_reskinned": int(len(reskinned_faces)),
        "same_vertex_shape": bool(original_vertices.shape == reskinned_vertices.shape),
        "same_face_shape": bool(original_faces.shape == reskinned_faces.shape),
        "faces_equal": bool(np.array_equal(original_faces, reskinned_faces)),
        "max_abs_vertex_delta_m": max_abs,
        "mean_abs_vertex_delta_m": mean_abs,
        "passed": bool(
            original_vertices.shape == reskinned_vertices.shape
            and original_faces.shape == reskinned_faces.shape
            and np.array_equal(original_faces, reskinned_faces)
            and max_abs <= 1e-12
        ),
    }


__all__ = [
    "SKIN_CHOICES",
    "ORIGINAL_MESH_COLOR",
    "RESKIN_EMISSIVE_COLOR",
    "RESKIN_FRESNEL_EXPONENT",
    "RESKIN_FRESNEL_STRENGTH",
    "make_render_material",
    "make_vertex_colors",
    "compare_same_geometry",
]
