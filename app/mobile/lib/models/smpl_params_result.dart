/// Decoded SMPL parameters from the server's compact binary response.
///
/// Binary layout (little-endian):
///   magic            : uint32  (0x534D504C = 'SMPL')
///   frame_count      : uint32
///   params_per_frame : uint32  (79)
///   fps              : float32
///   focal_length     : float32  (estimated pixels, 0 if unknown)
///   params           : frame_count × 79 × float32
///                        [go(3), body_pose(63), betas(10), cam_t(3)]
///   valid            : frame_count × uint8
library;

import 'dart:typed_data';

class SmplParamsResult {
  final int frameCount;
  final double fps;
  final double focalLength;

  /// Flat array: [frameCount × 79] float32
  final Float32List params;

  /// 1 = valid detection, 0 = no person detected
  final Uint8List valid;

  static const int paramsPerFrame = 79;
  static const int _magic = 0x534D504C;

  SmplParamsResult({
    required this.frameCount,
    required this.fps,
    required this.focalLength,
    required this.params,
    required this.valid,
  });

  factory SmplParamsResult.fromBinary(Uint8List bytes) {
    final bd = ByteData.sublistView(bytes);
    int offset = 0;

    final magic = bd.getUint32(offset, Endian.little); offset += 4;
    if (magic != _magic) {
      throw FormatException('Invalid SMPL binary magic: 0x${magic.toRadixString(16)}');
    }

    final frameCount = bd.getUint32(offset, Endian.little); offset += 4;
    final paramsPerFrameRead = bd.getUint32(offset, Endian.little); offset += 4;
    final fps = bd.getFloat32(offset, Endian.little); offset += 4;

    // focal_length added in v2 header (paramsPerFrame == 79).
    double focalLength = 0.0;
    if (paramsPerFrameRead == paramsPerFrame) {
      focalLength = bd.getFloat32(offset, Endian.little); offset += 4;
    } else {
      throw FormatException('Unexpected params_per_frame: $paramsPerFrameRead (expected $paramsPerFrame)');
    }

    final paramCount = frameCount * paramsPerFrame;
    final params = Float32List(paramCount);
    for (int i = 0; i < paramCount; i++) {
      params[i] = bd.getFloat32(offset, Endian.little);
      offset += 4;
    }

    final valid = bytes.sublist(offset, offset + frameCount);

    return SmplParamsResult(
      frameCount: frameCount,
      fps: fps,
      focalLength: focalLength,
      params: params,
      valid: valid,
    );
  }

  /// Extract go[3] for frame [i].
  List<double> go(int i) {
    final base = i * paramsPerFrame;
    return [params[base], params[base + 1], params[base + 2]];
  }

  /// Extract body_pose[63] for frame [i].
  List<double> bodyPose(int i) {
    final base = i * paramsPerFrame + 3;
    return List<double>.generate(63, (k) => params[base + k]);
  }

  /// Extract betas[10] for frame [i].
  List<double> betas(int i) {
    final base = i * paramsPerFrame + 66;
    return List<double>.generate(10, (k) => params[base + k]);
  }

  /// Extract cam_t[3] = [tx, ty, tz] camera-space translation for frame [i].
  List<double> camT(int i) {
    final base = i * paramsPerFrame + 76;
    return [params[base], params[base + 1], params[base + 2]];
  }

  bool isValid(int i) => valid[i] != 0;
}
