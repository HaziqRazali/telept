/// Decoded MHR parameters from the server's compact binary response.
///
/// Binary layout (little-endian):
///   magic            : uint32  (0x4D485250 = 'MHRP')
///   frame_count      : uint32
///   params_per_frame : uint32  (207)
///   fps              : float32
///   focal_length     : float32  (estimated pixels, 0 if unknown)
///   params           : frame_count × 207 × float32
///                        [model_params(204), cam_t(3)]
///   valid            : frame_count × uint8
///
/// model_params[204] layout:
///   [0:3]    global_trans * 10  (always ~0; root position is in cam_t)
///   [3:6]    global_rot  euler ZYX (radians)
///   [6:136]  body_pose   euler ZYX for 130 body joints (radians)
///   [136:204] scales     per-joint scale values (68 floats, from PCA decomp)
library;

import 'dart:typed_data';

class MhrParamsResult {
  final int frameCount;
  final double fps;
  final double focalLength;

  /// Flat array: [frameCount × 207] float32
  final Float32List params;

  /// 1 = valid detection, 0 = no person detected
  final Uint8List valid;

  static const int paramsPerFrame = 207;
  static const int modelParamsCount = 204;
  static const int _magic = 0x4D485250; // 'MHRP'

  MhrParamsResult({
    required this.frameCount,
    required this.fps,
    required this.focalLength,
    required this.params,
    required this.valid,
  });

  factory MhrParamsResult.fromBinary(Uint8List bytes) {
    final bd = ByteData.sublistView(bytes);
    int offset = 0;

    final magic = bd.getUint32(offset, Endian.little); offset += 4;
    if (magic != _magic) {
      throw FormatException('Invalid MHR binary magic: 0x${magic.toRadixString(16)}');
    }

    final frameCount = bd.getUint32(offset, Endian.little); offset += 4;
    final paramsPerFrameRead = bd.getUint32(offset, Endian.little); offset += 4;
    if (paramsPerFrameRead != paramsPerFrame) {
      throw FormatException(
          'Unexpected params_per_frame: $paramsPerFrameRead (expected $paramsPerFrame)');
    }
    final fps = bd.getFloat32(offset, Endian.little); offset += 4;
    final focalLength = bd.getFloat32(offset, Endian.little); offset += 4;

    final paramCount = frameCount * paramsPerFrame;
    final params = Float32List(paramCount);
    for (int i = 0; i < paramCount; i++) {
      params[i] = bd.getFloat32(offset, Endian.little);
      offset += 4;
    }

    final valid = Uint8List.fromList(bytes.sublist(offset, offset + frameCount));

    return MhrParamsResult(
      frameCount: frameCount,
      fps: fps,
      focalLength: focalLength,
      params: params,
      valid: valid,
    );
  }

  /// Extract model_params[204] for frame [i].
  Float32List modelParams(int i) {
    final base = i * paramsPerFrame;
    return Float32List.sublistView(params, base, base + modelParamsCount);
  }

  /// Extract cam_t[3] = [tx, ty, tz] for frame [i].
  List<double> camT(int i) {
    final base = i * paramsPerFrame + modelParamsCount;
    return [params[base], params[base + 1], params[base + 2]];
  }

  bool isValid(int i) => valid[i] != 0;
}
