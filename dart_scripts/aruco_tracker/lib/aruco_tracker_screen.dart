import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

// ─── Physical marker size (metres) ───────────────────────────────────────────
const double kMarkerSizeM = 0.05; // 5 cm — change to match your printed marker

// ─── ArUco dictionary ─────────────────────────────────────────────────────────
const cv.PredefinedDictionaryType kArucoDict =
    cv.PredefinedDictionaryType.DICT_4X4_50;

class ArucoTrackerScreen extends StatefulWidget {
  const ArucoTrackerScreen({super.key});

  @override
  State<ArucoTrackerScreen> createState() => _ArucoTrackerScreenState();
}

class _ArucoTrackerScreenState extends State<ArucoTrackerScreen> {
  // ── Shared state ─────────────────────────────────────────────────────────
  Uint8List? _frameBytes; // JPEG bytes rendered by Image.memory
  String _statusText = 'Initialising…';
  bool _running = false;

  // ── dartcv4 objects (all platforms) ──────────────────────────────────────
  late cv.ArucoDetector _detector;
  cv.Mat? _cameraMatrix;
  cv.Mat? _distCoeffs;

  // ── Linux-only ────────────────────────────────────────────────────────────
  cv.VideoCapture? _cap;

  // ── iOS/Android (camera plugin) ──────────────────────────────────────────
  CameraController? _camCtrl;
  bool _processingFrame = false; // guard: skip if previous frame still running

  // ─────────────────────────────────────────────────────────────────────────
  @override
  void initState() {
    super.initState();
    _fixSigprofRestart();
    _initDetector();
    _startCamera();
  }

  // In debug mode, the Dart VM's SIGPROF CPU-profiler fires every ~1ms.
  // siginterrupt(SIGPROF, 0) sets SA_RESTART so that blocked syscalls
  // (select/pselect used by OpenCV's V4L2 backend) are automatically
  // restarted after each SIGPROF, instead of returning EINTR → false.
  static void _fixSigprofRestart() {
    if (!Platform.isLinux) return;
    const int sigprof = 27;
    final libc = DynamicLibrary.open('libc.so.6');
    final siginterruptFn = libc.lookupFunction<
        Int32 Function(Int32, Int32),
        int Function(int, int)>('siginterrupt');
    siginterruptFn(sigprof, 0); // 0 = restart syscalls on signal (SA_RESTART)
  }

  @override
  void dispose() {
    _running = false;
    _cap?.release();
    _camCtrl?.dispose();
    _detector.dispose();
    _cameraMatrix?.dispose();
    _distCoeffs?.dispose();
    super.dispose();
  }

  // ── Initialise ArUco detector (same on all platforms) ────────────────────
  void _initDetector() {
    final dict = cv.ArucoDictionary.predefined(kArucoDict);
    final params = cv.ArucoDetectorParameters.empty();
    _detector = cv.ArucoDetector.create(dict, params);
    dict.dispose();
    params.dispose();
  }

  // ── Lazy-initialise camera intrinsic matrix from first frame size ─────────
  void _initCameraMatrix(int w, int h) {
    if (_cameraMatrix != null) return;
    final fx = (w > h ? w : h) * 0.7;
    _cameraMatrix = cv.Mat.fromList(3, 3, cv.MatType.CV_64FC1, [
      fx, 0.0, w / 2.0,
      0.0, fx, h / 2.0,
      0.0, 0.0, 1.0,
    ]);
    _distCoeffs = cv.Mat.fromList(4, 1, cv.MatType.CV_64FC1, [0.0, 0.0, 0.0, 0.0]);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // PLATFORM BRANCHING: Camera startup
  // ─────────────────────────────────────────────────────────────────────────
  Future<void> _startCamera() async {
    if (Platform.isLinux || Platform.isMacOS || Platform.isWindows) {
      await _startLinuxCamera();
    } else {
      await _startMobileCamera();
    }
  }

  // ── Linux / desktop: use dartcv4 VideoCapture ─────────────────────────────
  Future<void> _startLinuxCamera() async {
    debugPrint('[CAM] Opening VideoCapture.fromDevice(0)');
    _cap = cv.VideoCapture.fromDevice(0);
    debugPrint('[CAM] isOpened=${_cap!.isOpened}');

    _cap!.set(cv.CAP_PROP_FRAME_WIDTH, 640);
    _cap!.set(cv.CAP_PROP_FRAME_HEIGHT, 480);
    _cap!.set(cv.CAP_PROP_FPS, 30);
    debugPrint('[CAM] W=${_cap!.get(cv.CAP_PROP_FRAME_WIDTH)} H=${_cap!.get(cv.CAP_PROP_FRAME_HEIGHT)}');

    if (!_cap!.isOpened) {
      setState(() => _statusText = 'Cannot open camera 0');
      return;
    }

    // Give V4L2 time to fill its buffers after VIDIOC_STREAMON.
    // Without this, the first reads return EAGAIN (no frame ready yet).
    await Future.delayed(const Duration(milliseconds: 500));

    // Drain warm-up frames — dispose each Mat to avoid native memory leaks
    for (var i = 0; i < 10; i++) {
      final (ok, m) = _cap!.read();
      debugPrint('[CAM] warmup $i: grabbed=$ok empty=${m.isEmpty}');
      m.dispose();
    }

    _running = true;
    setState(() => _statusText = '');
    _linuxFrameLoop();
  }

  int _frameCount = 0; // diagnostic counter

  void _linuxFrameLoop() {
    if (!_running || !mounted) return;

    try {
      final (grabbed, frame) = _cap!.read();
      if (_frameCount < 3) {
        debugPrint('[LOOP] frame $_frameCount: grabbed=$grabbed empty=${frame.isEmpty} ${frame.cols}x${frame.rows}');
      }
      _frameCount++;
      if (grabbed && !frame.isEmpty) {
        Uint8List? jpegBytes;
        try {
          jpegBytes = _processAndAnnotate(frame);
        } catch (e) {
          // ArUco processing failed — fall back to raw JPEG so we can see the feed
          final (ok, buf) = cv.imencode('.jpg', frame);
          if (ok) jpegBytes = buf;
          if (mounted) setState(() => _statusText = 'Processing error: $e');
        }
        frame.dispose();
        if (mounted && jpegBytes != null) {
          setState(() {
            _frameBytes = jpegBytes;
            _statusText = ''; // clear any previous transient error
          });
        }
      }
      // If grabbed=false, we just skip silently and retry on next tick.
      // In O_NONBLOCK mode with OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT=0 this is
      // normal: DQBUF returns EAGAIN when no frame is ready yet.
    } catch (e) {
      if (mounted) setState(() => _statusText = 'Frame loop error: $e');
    }

    // 30ms ≈ 33fps cap — avoids hammering V4L2 DQBUF faster than frames arrive.
    // Duration.zero caused 100% EAGAIN returns in debug mode (O_NONBLOCK V4L2).
    Future.delayed(const Duration(milliseconds: 30), _linuxFrameLoop);
  }

  // ── Mobile (iOS / Android): use camera plugin ────────────────────────────
  Future<void> _startMobileCamera() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() => _statusText = 'Camera permission denied');
      return;
    }

    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      setState(() => _statusText = 'No cameras found');
      return;
    }

    // Prefer the back camera for AR use
    final back = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    _camCtrl = CameraController(
      back,
      ResolutionPreset.medium, // 720p — balance between quality and speed
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.bgra8888, // iOS default; consistent
    );

    await _camCtrl!.initialize();
    _running = true;
    setState(() => _statusText = '');
    _camCtrl!.startImageStream(_onMobileFrame);
  }

  void _onMobileFrame(CameraImage image) {
    if (_processingFrame || !_running) return;
    _processingFrame = true;

    // Convert CameraImage → cv.Mat, then process
    final mat = _cameraImageToMat(image);
    if (mat != null) {
      final jpegBytes = _processAndAnnotate(mat);
      mat.dispose();
      if (mounted) setState(() => _frameBytes = jpegBytes);
    }

    _processingFrame = false;
  }

  /// Convert a BGRA8888 CameraImage to a BGR cv.Mat.
  cv.Mat? _cameraImageToMat(CameraImage image) {
    try {
      final w = image.width;
      final h = image.height;
      final plane = image.planes[0];
      final rowStride = plane.bytesPerRow;

      Uint8List bgraBytes;
      if (rowStride == w * 4) {
        bgraBytes = plane.bytes;
      } else {
        // Strip row padding
        bgraBytes = Uint8List(w * h * 4);
        for (var r = 0; r < h; r++) {
          bgraBytes.setRange(r * w * 4, (r + 1) * w * 4, plane.bytes,
              r * rowStride);
        }
      }

      final bgraMat =
          cv.Mat.fromList(h, w, cv.MatType.CV_8UC4, bgraBytes);
      final bgrMat = cv.cvtColor(bgraMat, cv.COLOR_BGRA2BGR);
      bgraMat.dispose();
      return bgrMat;
    } catch (_) {
      return null;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CORE: Detect markers and draw overlays onto the frame Mat
  // Returns JPEG bytes for display.
  // ─────────────────────────────────────────────────────────────────────────
  Uint8List? _processAndAnnotate(cv.Mat frame) {
    final w = frame.cols;
    final h = frame.rows;
    _initCameraMatrix(w, h);

    // --- Detect markers ---
    final gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY);
    final (corners, ids, _) = _detector.detectMarkers(gray);
    gray.dispose();

    if (ids.isNotEmpty) {
      // Draw marker borders
      cv.arucoDrawDetectedMarkers(
          frame, corners, ids, cv.Scalar(0, 255, 0, 0));

      // 3D object points: marker corners in marker-local space (Z=0 plane)
      final h2 = kMarkerSizeM / 2;
      final objPts = cv.Mat.fromList(4, 1, cv.MatType.CV_32FC3, [
        -h2,  h2, 0.0, // top-left
         h2,  h2, 0.0, // top-right
         h2, -h2, 0.0, // bottom-right
        -h2, -h2, 0.0, // bottom-left
      ]);

      for (var i = 0; i < ids.length; i++) {
        final markerId = ids[i];
        final c = corners[i]; // VecPoint2f, 4 corners

        // Build (4,1,CV_32FC2) image-points matrix for solvePnP
        final imgPts = cv.Mat.fromList(4, 1, cv.MatType.CV_32FC2, [
          c[0].x.toDouble(), c[0].y.toDouble(),
          c[1].x.toDouble(), c[1].y.toDouble(),
          c[2].x.toDouble(), c[2].y.toDouble(),
          c[3].x.toDouble(), c[3].y.toDouble(),
        ]);

        final (ok, rvec, tvec) = cv.solvePnP(
          objPts,
          imgPts,
          _cameraMatrix!,
          _distCoeffs!,
        );
        imgPts.dispose();

        if (ok) {
          // Draw XYZ axes (X=red, Y=green, Z=blue)
          cv.drawFrameAxes(
            frame,
            _cameraMatrix!,
            _distCoeffs!,
            rvec,
            tvec,
            kMarkerSizeM * 0.8,
            thickness: 3,
          );

          // Distance label
          final tx = tvec.atF64(0, i1: 0);
          final ty = tvec.atF64(1, i1: 0);
          final tz = tvec.atF64(2, i1: 0);
          final distM = math.sqrt(tx * tx + ty * ty + tz * tz);

          // ID label above top-left corner
          cv.putText(
            frame,
            'ID $markerId',
            cv.Point(c[0].x.toInt(), c[0].y.toInt() - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            cv.Scalar(0, 255, 255, 0),
            thickness: 2,
          );

          // Distance label at marker centre
          final cx = ((c[0].x + c[2].x) / 2).toInt();
          final cy = ((c[0].y + c[2].y) / 2).toInt();
          cv.putText(
            frame,
            '${(distM * 100).toStringAsFixed(1)} cm',
            cv.Point(cx - 30, cy + 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            cv.Scalar(255, 255, 255, 0),
            thickness: 1,
          );

          rvec.dispose();
          tvec.dispose();
        }
      }

      objPts.dispose();
    } else {
      cv.putText(
        frame,
        'No markers detected',
        cv.Point(10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        cv.Scalar(0, 0, 200, 0),
        thickness: 2,
      );
    }

    final (_, jpegBytes) = cv.imencode('.jpg', frame);
    return jpegBytes;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // BUILD
  // ─────────────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: const Text('ArUco Tracker',
            style: TextStyle(color: Colors.white)),
      ),
      body: _statusText.isNotEmpty
          ? Center(
              child: Text(_statusText,
                  style: const TextStyle(color: Colors.white, fontSize: 18)))
          : _frameBytes == null
              ? const Center(child: CircularProgressIndicator())
              : SizedBox.expand(
                  child: Image.memory(
                    _frameBytes!,
                    gaplessPlayback: true, // prevents flicker between frames
                    fit: BoxFit.fill,
                  ),
                ),
    );
  }
}
