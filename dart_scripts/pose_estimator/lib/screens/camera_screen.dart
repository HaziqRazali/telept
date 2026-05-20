/// Main camera screen — routes to the correct platform implementation.
///
/// Linux : [_LinuxScreen]
///   Spawns mediapipe_bridge.py (Python / conda) as a subprocess.
///   Receives JPEG frames + 33 MediaPipe landmarks via stdout.
///
/// iOS / Android : [_MobileScreen]
///   Uses the [camera] package for live preview and
///   [google_mlkit_pose_detection] (BlazePose) for landmarks.
///   (On iOS the colleague's native Swift MediaPipe plugin will replace
///   google_mlkit_pose_detection when he merges the app.)
///
/// After landmark extraction the pipeline is identical on both platforms:
///   landmarks (33) → PoseEstimatorService (TFLite kpts2smpl) → 204 MHR params
///   → MhrModel.forward() → MeshFrame → MeshViewer
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

import '../models/mesh_frame.dart';
import '../services/linux_pose_service.dart';
import '../services/mhr_model.dart';
import '../services/pose_estimator_service.dart';
import '../widgets/mesh_viewer.dart';

// ── Router ────────────────────────────────────────────────────────────────────

class CameraScreen extends StatelessWidget {
  const CameraScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Platform.isLinux ? const _LinuxScreen() : const _MobileScreen();
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// Linux implementation
// ═════════════════════════════════════════════════════════════════════════════

class _LinuxScreen extends StatefulWidget {
  const _LinuxScreen();

  @override
  State<_LinuxScreen> createState() => _LinuxScreenState();
}

class _LinuxScreenState extends State<_LinuxScreen> {
  Uint8List? _jpeg;
  List<({double x, double y, double score})> _landmarks = [];
  MeshFrame _meshFrame = const MeshFrame(vertices: [], indices: []);
  int _imgW = 0;
  int _imgH = 0;

  @override
  void initState() {
    super.initState();
    _start();
  }

  Future<void> _start() async {
    await LinuxPoseService.instance.start();

    LinuxPoseService.instance.stream?.listen((frame) {
      if (!mounted) return;

      // body_params already computed + filtered by the Python bridge.
      final mhrParams = frame.bodyParams;
      // Normalise pixel landmarks to [0,1] for the skeleton painter.
      final displayLandmarks = [
        for (final lm in frame.landmarks)
          (x: lm.x / frame.imgW, y: lm.y / frame.imgH, score: lm.score),
      ];
      MeshFrame? meshFrame;
      if (mhrParams.any((v) => v != 0.0)) {
        meshFrame = MhrModel.instance.forward(modelParams: mhrParams);
      }

      setState(() {
        _jpeg = frame.jpeg;
        _imgW = frame.imgW;
        _imgH = frame.imgH;
        _landmarks = displayLandmarks;
        if (meshFrame != null) _meshFrame = meshFrame;
      });
    });
  }

  @override
  void dispose() {
    LinuxPoseService.instance.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: const Text('Pose Estimator'),
        centerTitle: true,
      ),
      body: Row(
        children: [
          // ── Left: camera frame + skeleton overlay ──────────────────────
          Expanded(
            flex: 1,
            child: _jpeg == null
                ? const Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        CircularProgressIndicator(color: Colors.white),
                        SizedBox(height: 12),
                        Text(
                          'Starting MediaPipe…',
                          style: TextStyle(color: Colors.white54),
                        ),
                      ],
                    ),
                  )
                : FittedBox(
                    fit: BoxFit.contain,
                    child: SizedBox(
                      width: _imgW.toDouble(),
                      height: _imgH.toDouble(),
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          Image.memory(
                            _jpeg!,
                            fit: BoxFit.fill,
                            gaplessPlayback: true,
                          ),
                          if (_landmarks.isNotEmpty)
                            CustomPaint(
                              painter: _SkeletonPainter(landmarks: _landmarks),
                            ),
                        ],
                      ),
                    ),
                  ),
          ),
          const VerticalDivider(width: 1, color: Colors.white24),
          // ── Right: 3D MHR mesh viewer ──────────────────────────────────
          Expanded(
            flex: 1,
            child: Container(
              color: const Color(0xFF1A1A2E),
              child: MeshViewer(frame: _meshFrame),
            ),
          ),
        ],
      ),
    );
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// Mobile (iOS / Android) implementation
// ═════════════════════════════════════════════════════════════════════════════

class _MobileScreen extends StatefulWidget {
  const _MobileScreen();

  @override
  State<_MobileScreen> createState() => _MobileScreenState();
}

class _MobileScreenState extends State<_MobileScreen>
    with WidgetsBindingObserver {
  CameraController? _cameraController;
  bool _cameraInitialized = false;

  final PoseDetector _poseDetector = PoseDetector(
    options: PoseDetectorOptions(mode: PoseDetectionMode.stream),
  );
  bool _isProcessing = false;

  List<({double x, double y, double score})> _landmarks = [];
  MeshFrame _meshFrame = const MeshFrame(vertices: [], indices: []);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) return;

    final camera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );

    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: Platform.isIOS
          ? ImageFormatGroup.bgra8888
          : ImageFormatGroup.yuv420,
    );

    await controller.initialize();
    if (!mounted) return;

    setState(() {
      _cameraController = controller;
      _cameraInitialized = true;
    });

    controller.startImageStream(_onCameraImage);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      controller.stopImageStream();
    } else if (state == AppLifecycleState.resumed) {
      controller.startImageStream(_onCameraImage);
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    _poseDetector.close();
    super.dispose();
  }

  void _onCameraImage(CameraImage image) {
    if (_isProcessing) return;
    _isProcessing = true;
    _processFrame(image).then((_) => _isProcessing = false);
  }

  Future<void> _processFrame(CameraImage image) async {
    final camera = _cameraController!.description;
    final rotation = Platform.isIOS
        ? (InputImageRotationValue.fromRawValue(camera.sensorOrientation) ??
            InputImageRotation.rotation0deg)
        : InputImageRotation.rotation0deg;

    final inputImage = InputImage.fromBytes(
      bytes: _concatenatePlanes(image.planes),
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: Platform.isIOS
            ? InputImageFormat.bgra8888
            : InputImageFormat.yuv_420_888,
        bytesPerRow: image.planes.first.bytesPerRow,
      ),
    );

    final poses = await _poseDetector.processImage(inputImage);
    if (!mounted || poses.isEmpty) return;

    final pose = poses.first;
    // Pixel-space landmarks for inference (ML Kit reports x/y in pixels).
    final pixelLandmarks = List<({double x, double y, double score})>.generate(
      33,
      (i) {
        final lm = pose.landmarks[PoseLandmarkType.values[i]];
        if (lm == null) return (x: 0.0, y: 0.0, score: 0.0);
        return (x: lm.x.toDouble(), y: lm.y.toDouble(), score: lm.likelihood.toDouble());
      },
    );
    // [0,1] normalised landmarks for the skeleton painter.
    final displayLandmarks = [
      for (final lm in pixelLandmarks)
        (
          x: (lm.x / image.width).clamp(0.0, 1.0),
          y: (lm.y / image.height).clamp(0.0, 1.0),
          score: lm.score,
        ),
    ];

    final mhrParams = PoseEstimatorService.instance.run(pixelLandmarks);
    if (mhrParams == null) return;

    final frame = MhrModel.instance.forward(modelParams: mhrParams);
    if (mounted) {
      setState(() {
        _landmarks = displayLandmarks;
        _meshFrame = frame;
      });
    }
  }

  static Uint8List _concatenatePlanes(List<Plane> planes) {
    final bytes = <int>[];
    for (final plane in planes) {
      bytes.addAll(plane.bytes);
    }
    return Uint8List.fromList(bytes);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: const Text('Pose Estimator'),
        centerTitle: true,
      ),
      body: Column(
        children: [
          Expanded(
            flex: 1,
            child: _cameraInitialized
                ? Stack(
                    fit: StackFit.expand,
                    children: [
                      CameraPreview(_cameraController!),
                      if (_landmarks.isNotEmpty)
                        CustomPaint(
                          painter: _SkeletonPainter(landmarks: _landmarks),
                        ),
                    ],
                  )
                : const Center(
                    child: CircularProgressIndicator(color: Colors.white),
                  ),
          ),
          const Divider(height: 1, color: Colors.white24),
          Expanded(
            flex: 1,
            child: Container(
              color: const Color(0xFF1A1A2E),
              child: MeshViewer(frame: _meshFrame),
            ),
          ),
        ],
      ),
    );
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// Shared skeleton painter — landmarks normalised to [0, 1] on both platforms
// ═════════════════════════════════════════════════════════════════════════════

class _SkeletonPainter extends CustomPainter {
  final List<({double x, double y, double score})> landmarks;

  static const List<(int, int)> _connections = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (27, 31), (28, 32),
  ];

  const _SkeletonPainter({required this.landmarks});

  @override
  void paint(Canvas canvas, Size size) {
    if (landmarks.length < 33) return;

    final bonePaint = Paint()
      ..color = Colors.greenAccent.withValues(alpha: 0.8)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    final dotPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.fill;

    for (final (a, b) in _connections) {
      final la = landmarks[a];
      final lb = landmarks[b];
      if (la.score < 0.3 || lb.score < 0.3) continue;
      canvas.drawLine(
        Offset(la.x * size.width, la.y * size.height),
        Offset(lb.x * size.width, lb.y * size.height),
        bonePaint,
      );
    }

    for (final idx in PoseEstimatorService.kptsIndices) {
      final lm = landmarks[idx];
      if (lm.score < 0.3) continue;
      canvas.drawCircle(
        Offset(lm.x * size.width, lm.y * size.height),
        4.0,
        dotPaint,
      );
    }
  }

  @override
  bool shouldRepaint(_SkeletonPainter old) => old.landmarks != landmarks;
}
