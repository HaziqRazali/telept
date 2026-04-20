/// Decoded SMPL parameters from the server's compact binary response.
///
/// Binary layout (little-endian):
///   magic       : uint32  (0x534D504C = 'SMPL')
///   frame_count : uint32
///   params_per_frame : uint32  (76)
///   fps         : float32
///   params      : frame_count × 76 × float32
///                   [go(3), body_pose(63), betas(10)]
///   valid       : frame_count × uint8
library;

import 'dart:typed_data';

class SmplParamsResult {
  final int frameCount;
  final double fps;

  /// Flat array: [frameCount × 76] float32
  /// Frame i: params[i*76 .. i*76+76]
  ///   go:        params[i*76 +  0 ..  2]  (3 floats, axis-angle)
  ///   body_pose: params[i*76 +  3 .. 65]  (63 floats)
  ///   betas:     params[i*76 + 66 .. 75]  (10 floats)
  final Float32List params;

  /// 1 = valid detection, 0 = no person detected
  final Uint8List valid;

  static const int paramsPerFrame = 76;
  static const int _magic = 0x534D504C;

  SmplParamsResult({
    required this.frameCount,
    required this.fps,
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

    if (paramsPerFrameRead != paramsPerFrame) {
      throw FormatException('Unexpected params_per_frame: $paramsPerFrameRead');
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

  bool isValid(int i) => valid[i] != 0;
}
