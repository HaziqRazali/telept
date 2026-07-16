/// TFLite-based pose estimator service.
///
/// Pipeline (mirrors the Python video_inference.py pipeline exactly):
///   1. Receive 33 MediaPipe landmarks in **pixel** coordinates.
///   2. Torso-span normalise:
///        origin    = mid-hip  (landmarks 23 + 24)
///        scale     = ‖mid-shoulder − mid-hip‖  (landmarks 11 + 12)
///        kpts_norm = (kpts_px − origin) / scale
///   3. Select the 21 mediapipe_full_body keypoints.
///   4. Zero joints whose visibility ≤ kScoreThr (matches training pipeline).
///   5. Run TFLite inference → 130 floats (MHR body_pose_euler).
///   6. Apply per-dimension One Euro filter (min_cutoff=0.5, β=0.02).
///   7. Pad to 204 floats for MhrModel.forward().
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';

// ── One Euro low-pass filter ──────────────────────────────────────────────────

class _OneEuro {
  final double minCutoff;
  final double beta;
  static const double _dCutoff = 1.0;

  double? _xFilt;
  double? _dxFilt;
  int?    _lastMs;

  _OneEuro({this.minCutoff = 0.5, this.beta = 0.02});

  static double _alpha(double cutoff, double dt) {
    final tau = 1.0 / (2.0 * math.pi * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  double filter(double x, int nowMs) {
    final dt = (_lastMs == null)
        ? (1.0 / 30.0)
        : math.max((nowMs - _lastMs!) / 1000.0, 1e-6);
    _lastMs = nowMs;

    final dx  = (_xFilt  == null) ? 0.0 : (x - _xFilt!) / dt;
    final aD  = _alpha(_dCutoff, dt);
    _dxFilt   = (_dxFilt == null) ? dx : _dxFilt! + aD * (dx - _dxFilt!);

    final cutoff = minCutoff + beta * _dxFilt!.abs();
    final a      = _alpha(cutoff, dt);
    _xFilt       = (_xFilt == null) ? x : _xFilt! + a * (x - _xFilt!);
    return _xFilt!;
  }

  void reset() { _xFilt = null; _dxFilt = null; _lastMs = null; }
}

// ── Pose estimator service ────────────────────────────────────────────────────

class PoseEstimatorService {
  // ── Constants ──────────────────────────────────────────────────────────────

  /// Subset of the 33 MediaPipe landmarks fed to the model
  /// (mediapipe_full_body_original_indices).
  static const List<int> kptsIndices = [
    0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32,
  ];

  /// Joints below this visibility are zeroed before inference.
  static const double kScoreThr = 0.5;

  // Torso anchor indices in the full 33-landmark MediaPipe space.
  static const int _kLHip = 23, _kRHip = 24;
  static const int _kLSho = 11, _kRSho = 12;

  /// Minimum torso span (pixels) — prevents division by near-zero.
  static const double _kMinSpan = 10.0;

  static const int _inputDim  = 42;   // 21 kpts × 2 (x, y)
  static const int _outputDim = 130;  // MHR body_pose_euler

  // ── State ──────────────────────────────────────────────────────────────────

  Interpreter? _interpreter;
  bool _loaded = false;

  /// One Euro filter — one instance per output dimension.
  final List<_OneEuro> _filters = List.generate(
    _outputDim,
    (_) => _OneEuro(minCutoff: 0.5, beta: 0.02),
  );

  static PoseEstimatorService? _instance;
  static PoseEstimatorService get instance =>
      _instance ??= PoseEstimatorService._();
  PoseEstimatorService._();

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  Future<void> load() async {
    if (_loaded) return;
    _interpreter = await Interpreter.fromAsset('assets/pose_model.tflite');
    _interpreter!.allocateTensors();
    _loaded = true;
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _loaded = false;
    for (final f in _filters) f.reset();
  }

  // ── Inference ──────────────────────────────────────────────────────────────

  /// Run inference on 33 MediaPipe landmarks given in **pixel coordinates**.
  ///
  /// Returns a [Float32List] of 204 floats ready for [MhrModel.forward()],
  /// or null when inference cannot proceed (not loaded, wrong count, no
  /// visible joints).
  Float32List? run(List<({double x, double y, double score})>? landmarks) {
    if (!_loaded || _interpreter == null) return null;
    if (landmarks == null || landmarks.length != 33) return null;

    // ── 1. Torso-span normalisation ────────────────────────────────────────
    final hipOk = landmarks[_kLHip].score > kScoreThr &&
                  landmarks[_kRHip].score > kScoreThr;
    final shoOk = landmarks[_kLSho].score > kScoreThr &&
                  landmarks[_kRSho].score > kScoreThr;

    double originX, originY, torsoSpan;

    if (hipOk && shoOk) {
      originX = 0.5 * (landmarks[_kLHip].x + landmarks[_kRHip].x);
      originY = 0.5 * (landmarks[_kLHip].y + landmarks[_kRHip].y);
      final mShoX = 0.5 * (landmarks[_kLSho].x + landmarks[_kRSho].x);
      final mShoY = 0.5 * (landmarks[_kLSho].y + landmarks[_kRSho].y);
      final dx = mShoX - originX, dy = mShoY - originY;
      torsoSpan = math.max(math.sqrt(dx * dx + dy * dy), _kMinSpan);
    } else {
      // Fallback: bounding box of all visible joints.
      double minX = double.infinity,  minY = double.infinity;
      double maxX = double.negativeInfinity, maxY = double.negativeInfinity;
      for (final lm in landmarks) {
        if (lm.score > kScoreThr) {
          if (lm.x < minX) minX = lm.x;
          if (lm.x > maxX) maxX = lm.x;
          if (lm.y < minY) minY = lm.y;
          if (lm.y > maxY) maxY = lm.y;
        }
      }
      if (!minX.isFinite) return null; // no visible joints at all
      originX   = minX;
      originY   = minY;
      torsoSpan = math.max(math.max(maxX - minX, maxY - minY), _kMinSpan);
    }

    // ── 2. Build input: select 21-kpt subset, normalise, zero low-score ───
    final input = Float32List(_inputDim);
    for (int i = 0; i < kptsIndices.length; i++) {
      final lm = landmarks[kptsIndices[i]];
      if (lm.score > kScoreThr) {
        input[i * 2]     = (lm.x - originX) / torsoSpan;
        input[i * 2 + 1] = (lm.y - originY) / torsoSpan;
      }
      // else: leave as 0.0  (matches training pipeline zeroing)
    }

    // ── 3. TFLite inference — input [1, 42] → output [1, 130] ─────────────
    // outputTensor is List<List<double>> shape [1][130]; tflite_flutter fills
    // it in-place via copyTo().
    final outputTensor = [List<double>.filled(_outputDim, 0.0)];
    _interpreter!.run([input.toList()], outputTensor);

    // ── 4. One Euro filter ─────────────────────────────────────────────────
    final nowMs    = DateTime.now().millisecondsSinceEpoch;
    final bodyPose = Float32List(_outputDim);
    for (int i = 0; i < _outputDim; i++) {
      bodyPose[i] = _filters[i].filter(outputTensor[0][i], nowMs);
    }

    // ── 5. Pad to 204-param MHR format ────────────────────────────────────
    //   [0:6]    global_trans*10 + global_rot_euler → 0
    //   [6:136]  body_pose_euler                    ← bodyPose
    //   [136:204] scales                            → 0
    final params = Float32List(204);
    for (int i = 0; i < 130; i++) {
      params[6 + i] = bodyPose[i];
    }
    return params;
  }
}
