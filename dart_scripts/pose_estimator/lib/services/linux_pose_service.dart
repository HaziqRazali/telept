/// Linux-only pose service.
///
/// Spawns [mediapipe_bridge.py] as a subprocess via conda and streams
/// [LinuxFrame] objects containing a JPEG frame + 33 MediaPipe landmarks +
/// 204 MHR body params (inference + One Euro filter already applied in Python).
///
/// Binary frame protocol (little-endian):
///   [0:4]      uint32   jpeg_len
///   [4:6]      uint16   frame_width
///   [6:8]      uint16   frame_height
///   [8:404]    float32[99]  landmarks [x0,y0,vis0,…,x32,y32,vis32] — PIXEL coords
///   [404:1220] float32[204] body_params — MHR 204-param array
///   [1220:]    jpeg_len bytes  JPEG data
library;

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';

// ── Constants ─────────────────────────────────────────────────────────────────

const int _kLmFloats    = 99;                              // 33 landmarks × 3
const int _kBodyFloats  = 204;                             // MHR body params
const int _kHeaderSize  = 4 + 2 + 2 + _kLmFloats * 4 + _kBodyFloats * 4; // = 1220

// ── Data class ────────────────────────────────────────────────────────────────

class LinuxFrame {
  /// JPEG-encoded camera frame (for display with Image.memory).
  final Uint8List jpeg;

  /// 33 MediaPipe landmarks, x/y in **pixel** coordinates.
  final List<({double x, double y, double score})> landmarks;

  /// Frame dimensions in pixels.
  final int imgW;
  final int imgH;

  /// 204-param MHR body array, already filtered — ready for MhrModel.forward().
  final Float32List bodyParams;

  const LinuxFrame({
    required this.jpeg,
    required this.landmarks,
    required this.imgW,
    required this.imgH,
    required this.bodyParams,
  });
}

// ── Service ───────────────────────────────────────────────────────────────────

class LinuxPoseService {
  Process? _process;
  StreamController<LinuxFrame>? _controller;
  StreamSubscription<List<int>>? _sub;

  /// Accumulates bytes until a complete frame has arrived.
  Uint8List _buf = Uint8List(0);

  static LinuxPoseService? _instance;
  static LinuxPoseService get instance =>
      _instance ??= LinuxPoseService._();
  LinuxPoseService._();

  Stream<LinuxFrame>? get stream => _controller?.stream;

  // ── Lifecycle ───────────────────────────────────────────────────────────────

  Future<void> start() async {
    final bytes =
        await rootBundle.load('assets/linux_bridge/mediapipe_bridge.py');
    final script = File('${Directory.systemTemp.path}/mediapipe_bridge.py');
    await script.writeAsBytes(bytes.buffer.asUint8List());

    // Extract TFLite model to a temp file so the Python bridge can load it.
    final modelBytes = await rootBundle.load('assets/pose_model.tflite');
    final modelFile  = File('${Directory.systemTemp.path}/pose_model.tflite');
    await modelFile.writeAsBytes(modelBytes.buffer.asUint8List());

    _buf = Uint8List(0);
    _controller = StreamController<LinuxFrame>.broadcast();

    _process = await Process.start(
      '/home/haziq/anaconda3/envs/pytorch_env/bin/python',
      [script.path, modelFile.path],
    );

    _sub = _process!.stdout.listen(
      _onBytes,
      onError: (_) {},
      onDone: _controller!.close,
    );

    _process!.stderr
        .transform(const SystemEncoding().decoder)
        .listen((s) => stderr.write(s));
  }

  Future<void> stop() async {
    await _sub?.cancel();
    _process?.kill();
    _process = null;
    await _controller?.close();
    _controller = null;
    _buf = Uint8List(0);
  }

  // ── Binary frame parser ───────────────────────────────────────────────────

  void _onBytes(List<int> incoming) {
    final combined = Uint8List(_buf.length + incoming.length);
    combined.setAll(0, _buf);
    combined.setAll(_buf.length, incoming);
    _buf = combined;
    _parseFrames();
  }

  void _parseFrames() {
    while (true) {
      if (_buf.length < _kHeaderSize) return;

      final bd      = ByteData.sublistView(_buf);
      final jpegLen = bd.getUint32(0, Endian.little);
      final frameLen = _kHeaderSize + jpegLen;

      if (_buf.length < frameLen) return; // incomplete frame — wait for more bytes

      final w = bd.getUint16(4, Endian.little);
      final h = bd.getUint16(6, Endian.little);

      // Parse 33 landmarks: each takes 12 bytes (3 × float32), starting at offset 8.
      final landmarks = List<({double x, double y, double score})>.generate(
        33,
        (i) {
          final base = 8 + i * 12;
          return (
            x:     bd.getFloat32(base,      Endian.little).toDouble(),
            y:     bd.getFloat32(base + 4,  Endian.little).toDouble(),
            score: bd.getFloat32(base + 8,  Endian.little).toDouble(),
          );
        },
      );

      // Parse 204 body params starting at offset 404 (after landmarks).
      const int _kBpOffset = 8 + _kLmFloats * 4; // = 404
      final bodyParams = Float32List(_kBodyFloats);
      for (int i = 0; i < _kBodyFloats; i++) {
        bodyParams[i] = bd.getFloat32(_kBpOffset + i * 4, Endian.little);
      }

      final jpeg = _buf.sublist(_kHeaderSize, frameLen); // Uint8List copy

      if (_controller != null && !_controller!.isClosed) {
        _controller!.add(LinuxFrame(
          jpeg: jpeg,
          landmarks: landmarks,
          imgW: w,
          imgH: h,
          bodyParams: bodyParams,
        ));
      }

      _buf = _buf.sublist(frameLen); // advance past this frame
    }
  }
}
