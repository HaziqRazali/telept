#!/usr/bin/env python3
"""MediaPipe pose detection + kpts2smpl inference bridge for Flutter/Linux.

Binary frame protocol (little-endian), one frame per write:

    [0:4]     uint32   jpeg_len
    [4:6]     uint16   frame_width
    [6:8]     uint16   frame_height
    [8:404]   float32[99]   landmarks — [x0,y0,vis0, …, x32,y32,vis32]
                             x, y are in PIXELS (not normalised to [0,1])
    [404:1220] float32[204] body_params — MHR 204-param array
    [1220:]   jpeg_len bytes  JPEG data

Usage: python mediapipe_bridge.py <path_to_pose_model.tflite>
"""

import os
import sys
import struct
import math
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_LM_FLOATS   = 99            # 33 landmarks × 3 (x, y, visibility)
_BODY_FLOATS = 204           # MHR body params
_HEADER_SIZE = 4 + 2 + 2 + _LM_FLOATS * 4 + _BODY_FLOATS * 4  # = 1220 bytes

# Subset of 33 MediaPipe landmarks fed to kpts2smpl (mediapipe_full_body indices)
_KPTS_INDICES = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24,
                 25, 26, 27, 28, 29, 30, 31, 32]
_SCORE_THR   = 0.5
_K_L_HIP, _K_R_HIP = 23, 24
_K_L_SHO, _K_R_SHO = 11, 12
_K_MIN_SPAN  = 10.0
_INPUT_DIM   = 42    # 21 kpts × 2
_OUTPUT_DIM  = 130   # kpts2smpl output

_TASK_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)
_TASK_MODEL_CACHE = os.path.expanduser(
    "~/.cache/mediapipe/pose_landmarker_full.task"
)


# ── MediaPipe model setup ────────────────────────────────────────────────────

def _ensure_task_model() -> str:
    if os.path.exists(_TASK_MODEL_CACHE):
        return _TASK_MODEL_CACHE
    os.makedirs(os.path.dirname(_TASK_MODEL_CACHE), exist_ok=True)
    print("Downloading MediaPipe pose landmarker model (~5 MB)…",
          file=sys.stderr, flush=True)
    urllib.request.urlretrieve(_TASK_MODEL_URL, _TASK_MODEL_CACHE)
    print("Done.", file=sys.stderr, flush=True)
    return _TASK_MODEL_CACHE


# ── One Euro low-pass filter ─────────────────────────────────────────────────

class _OneEuroFilter:
    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.02,
                 d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x_filt    = None
        self._dx_filt   = None
        self._last_time = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float) -> float:
        now = time.monotonic()
        dt  = (1.0 / 30.0) if self._last_time is None \
              else max(now - self._last_time, 1e-6)
        self._last_time = now

        dx            = 0.0 if self._x_filt is None else (x - self._x_filt) / dt
        a_d           = self._alpha(self.d_cutoff, dt)
        self._dx_filt = dx if self._dx_filt is None \
                        else self._dx_filt + a_d * (dx - self._dx_filt)

        cutoff       = self.min_cutoff + self.beta * abs(self._dx_filt)
        a            = self._alpha(cutoff, dt)
        self._x_filt = x if self._x_filt is None \
                       else self._x_filt + a * (x - self._x_filt)
        return self._x_filt


# ── kpts2smpl helpers ────────────────────────────────────────────────────────

def _make_interpreter(model_path: str):
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=model_path)
    except ImportError:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def _preprocess(lm_flat: list) -> "np.ndarray | None":
    """Torso-span normalise and select 21-kpt subset. Returns float32[42] or None."""
    def sx(i): return lm_flat[i * 3]
    def sy(i): return lm_flat[i * 3 + 1]
    def sc(i): return lm_flat[i * 3 + 2]

    hip_ok = sc(_K_L_HIP) > _SCORE_THR and sc(_K_R_HIP) > _SCORE_THR
    sho_ok = sc(_K_L_SHO) > _SCORE_THR and sc(_K_R_SHO) > _SCORE_THR

    if hip_ok and sho_ok:
        origin_x   = 0.5 * (sx(_K_L_HIP) + sx(_K_R_HIP))
        origin_y   = 0.5 * (sy(_K_L_HIP) + sy(_K_R_HIP))
        m_sho_x    = 0.5 * (sx(_K_L_SHO) + sx(_K_R_SHO))
        m_sho_y    = 0.5 * (sy(_K_L_SHO) + sy(_K_R_SHO))
        dx         = m_sho_x - origin_x
        dy         = m_sho_y - origin_y
        torso_span = max(math.sqrt(dx * dx + dy * dy), _K_MIN_SPAN)
    else:
        visible = [(sx(i), sy(i)) for i in range(33) if sc(i) > _SCORE_THR]
        if not visible:
            return None
        xs         = [p[0] for p in visible]
        ys         = [p[1] for p in visible]
        origin_x   = min(xs)
        origin_y   = min(ys)
        torso_span = max(max(max(xs) - origin_x, max(ys) - origin_y), _K_MIN_SPAN)

    inp = np.zeros(_INPUT_DIM, dtype=np.float32)
    for i, idx in enumerate(_KPTS_INDICES):
        if sc(idx) > _SCORE_THR:
            inp[i * 2]     = (sx(idx) - origin_x) / torso_span
            inp[i * 2 + 1] = (sy(idx) - origin_y) / torso_span
    return inp


def _run_inference(interp, input_details, output_details,
                   inp: "np.ndarray", filters: list) -> list:
    """Run kpts2smpl → One Euro filter → pad to 204-param MHR format."""
    interp.set_tensor(input_details[0]['index'], inp.reshape(1, _INPUT_DIM))
    interp.invoke()
    raw = interp.get_tensor(output_details[0]['index'])[0]  # (130,)

    filtered = [filters[i].filter(float(raw[i])) for i in range(_OUTPUT_DIM)]

    # MHR format: [0:6] zeros, [6:136] body_pose_euler, [136:204] zeros
    params = [0.0] * 204
    for i in range(130):
        params[6 + i] = filtered[i]
    return params


# ── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: mediapipe_bridge.py <kpts2smpl_model_path>",
              file=sys.stderr, flush=True)
        sys.exit(1)

    kpts2smpl_path = sys.argv[1]
    interp         = _make_interpreter(kpts2smpl_path)
    input_details  = interp.get_input_details()
    output_details = interp.get_output_details()
    filters        = [_OneEuroFilter(min_cutoff=0.5, beta=0.02)
                      for _ in range(_OUTPUT_DIM)]

    task_model_path = _ensure_task_model()

    base_opts = mp_python.BaseOptions(model_asset_path=task_model_path)
    lm_opts   = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open camera", file=sys.stderr, flush=True)
        sys.exit(1)

    out       = sys.stdout.buffer   # binary stdout
    start_ns  = time.monotonic_ns()

    with mp_vision.PoseLandmarker.create_from_options(lm_opts) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w        = frame.shape[:2]
            rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = (time.monotonic_ns() - start_ns) // 1_000_000

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect_for_video(mp_image, timestamp_ms)

            # ── Landmarks ────────────────────────────────────────────────────
            if result.pose_landmarks:
                lm_flat = []
                for lm in result.pose_landmarks[0]:  # first detected person
                    vis = lm.visibility if lm.visibility is not None else 0.0
                    lm_flat.extend([lm.x * w, lm.y * h, vis])
            else:
                lm_flat = [0.0] * _LM_FLOATS

            # ── kpts2smpl inference ──────────────────────────────────────────
            inp = _preprocess(lm_flat)
            if inp is not None:
                body_params = _run_inference(
                    interp, input_details, output_details, inp, filters)
            else:
                body_params = [0.0] * 204

            # ── JPEG encode ──────────────────────────────────────────────────
            _, jpeg_buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            jpeg_bytes = jpeg_buf.tobytes()

            # ── Write binary frame ───────────────────────────────────────────
            header   = struct.pack("<IHH", len(jpeg_bytes), w, h)
            lm_bytes = struct.pack(f"<{_LM_FLOATS}f", *lm_flat)
            bp_bytes = struct.pack("<204f", *body_params)
            out.write(header + lm_bytes + bp_bytes + jpeg_bytes)
            out.flush()

    cap.release()


if __name__ == "__main__":
    main()

_LM_FLOATS   = 99            # 33 landmarks × 3 (x, y, visibility)
_BODY_FLOATS = 204           # MHR body params
_HEADER_SIZE = 4 + 2 + 2 + _LM_FLOATS * 4 + _BODY_FLOATS * 4  # = 1220 bytes

# Subset of 33 MediaPipe landmarks fed to kpts2smpl (mediapipe_full_body indices)
_KPTS_INDICES = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24,
                 25, 26, 27, 28, 29, 30, 31, 32]
_SCORE_THR   = 0.5
_K_L_HIP, _K_R_HIP = 23, 24
_K_L_SHO, _K_R_SHO = 11, 12
_K_MIN_SPAN  = 10.0
_INPUT_DIM   = 42    # 21 kpts × 2
_OUTPUT_DIM  = 130   # kpts2smpl output


# ── One Euro low-pass filter ─────────────────────────────────────────────────

class _OneEuroFilter:
    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.02,
                 d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x_filt    = None
        self._dx_filt   = None
        self._last_time = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float) -> float:
        now = time.monotonic()
        dt  = (1.0 / 30.0) if self._last_time is None \
              else max(now - self._last_time, 1e-6)
        self._last_time = now

        dx            = 0.0 if self._x_filt is None else (x - self._x_filt) / dt
        a_d           = self._alpha(self.d_cutoff, dt)
        self._dx_filt = dx if self._dx_filt is None \
                        else self._dx_filt + a_d * (dx - self._dx_filt)

        cutoff       = self.min_cutoff + self.beta * abs(self._dx_filt)
        a            = self._alpha(cutoff, dt)
        self._x_filt = x if self._x_filt is None \
                       else self._x_filt + a * (x - self._x_filt)
        return self._x_filt


# ── kpts2smpl helpers ────────────────────────────────────────────────────────

def _make_interpreter(model_path: str):
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=model_path)
    except ImportError:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def _preprocess(lm_flat: list) -> "np.ndarray | None":
    """Torso-span normalise and select 21-kpt subset. Returns float32[42] or None."""
    def sx(i): return lm_flat[i * 3]
    def sy(i): return lm_flat[i * 3 + 1]
    def sc(i): return lm_flat[i * 3 + 2]

    hip_ok = sc(_K_L_HIP) > _SCORE_THR and sc(_K_R_HIP) > _SCORE_THR
    sho_ok = sc(_K_L_SHO) > _SCORE_THR and sc(_K_R_SHO) > _SCORE_THR

    if hip_ok and sho_ok:
        origin_x   = 0.5 * (sx(_K_L_HIP) + sx(_K_R_HIP))
        origin_y   = 0.5 * (sy(_K_L_HIP) + sy(_K_R_HIP))
        m_sho_x    = 0.5 * (sx(_K_L_SHO) + sx(_K_R_SHO))
        m_sho_y    = 0.5 * (sy(_K_L_SHO) + sy(_K_R_SHO))
        dx         = m_sho_x - origin_x
        dy         = m_sho_y - origin_y
        torso_span = max(math.sqrt(dx * dx + dy * dy), _K_MIN_SPAN)
    else:
        visible = [(sx(i), sy(i)) for i in range(33) if sc(i) > _SCORE_THR]
        if not visible:
            return None
        xs         = [p[0] for p in visible]
        ys         = [p[1] for p in visible]
        origin_x   = min(xs)
        origin_y   = min(ys)
        torso_span = max(max(max(xs) - origin_x, max(ys) - origin_y), _K_MIN_SPAN)

    inp = np.zeros(_INPUT_DIM, dtype=np.float32)
    for i, idx in enumerate(_KPTS_INDICES):
        if sc(idx) > _SCORE_THR:
            inp[i * 2]     = (sx(idx) - origin_x) / torso_span
            inp[i * 2 + 1] = (sy(idx) - origin_y) / torso_span
    return inp


def _run_inference(interp, input_details, output_details,
                   inp: "np.ndarray", filters: list) -> list:
    """Run kpts2smpl → One Euro filter → pad to 204-param MHR format."""
    interp.set_tensor(input_details[0]['index'], inp.reshape(1, _INPUT_DIM))
    interp.invoke()
    raw = interp.get_tensor(output_details[0]['index'])[0]  # (130,)

    filtered = [filters[i].filter(float(raw[i])) for i in range(_OUTPUT_DIM)]

    # MHR format: [0:6] zeros, [6:136] body_pose_euler, [136:204] zeros
    params = [0.0] * 204
    for i in range(130):
        params[6 + i] = filtered[i]
    return params


# ── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: mediapipe_bridge.py <model_path>", file=sys.stderr, flush=True)
        sys.exit(1)

    model_path     = sys.argv[1]
    interp         = _make_interpreter(model_path)
    input_details  = interp.get_input_details()
    output_details = interp.get_output_details()
    filters        = [_OneEuroFilter(min_cutoff=0.5, beta=0.02)
                      for _ in range(_OUTPUT_DIM)]

    mp_pose = mp.solutions.pose
    cap     = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open camera", file=sys.stderr, flush=True)
        sys.exit(1)

    out = sys.stdout.buffer   # binary stdout

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            _, jpeg_buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            jpeg_bytes = jpeg_buf.tobytes()

            # ── Landmarks ────────────────────────────────────────────────────
            if results.pose_landmarks:
                lm_flat = []
                for lm in results.pose_landmarks.landmark:
                    lm_flat.extend([lm.x * w, lm.y * h, lm.visibility])
            else:
                lm_flat = [0.0] * _LM_FLOATS

            # ── kpts2smpl inference ──────────────────────────────────────────
            inp = _preprocess(lm_flat)
            if inp is not None:
                body_params = _run_inference(
                    interp, input_details, output_details, inp, filters)
            else:
                body_params = [0.0] * 204

            # ── Write binary frame ───────────────────────────────────────────
            header   = struct.pack("<IHH", len(jpeg_bytes), w, h)
            lm_bytes = struct.pack(f"<{_LM_FLOATS}f", *lm_flat)
            bp_bytes = struct.pack("<204f", *body_params)
            out.write(header + lm_bytes + bp_bytes + jpeg_bytes)
            out.flush()

    cap.release()


if __name__ == "__main__":
    main()
