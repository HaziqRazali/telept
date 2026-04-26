import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import '../models/sensor_sample.dart';
import '../services/ble_service.dart';

/// Returned when the user finishes recording.
class RecordResult {
  final String videoPath;

  /// Null when BLE was not connected at recording start.
  final List<SensorSample>? sensorTimeline;

  const RecordResult({required this.videoPath, this.sensorTimeline});
}

/// Full-screen custom camera screen that simultaneously starts BLE capture.
class RecordScreen extends StatefulWidget {
  const RecordScreen({super.key});

  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  bool _isRecording = false;
  bool _cameraReady = false;
  String? _errorMsg;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      ctrl.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _initCamera() async {
    // Request permissions first
    await [Permission.camera, Permission.microphone].request();

    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      setState(() => _errorMsg = 'No camera found on this device.');
      return;
    }

    // Prefer the back camera
    final desc = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    final ctrl = CameraController(
      desc,
      ResolutionPreset.high,
      enableAudio: true,
    );

    try {
      await ctrl.initialize();
      if (!mounted) return;
      setState(() {
        _controller = ctrl;
        _cameraReady = true;
      });
    } on CameraException catch (e) {
      setState(() => _errorMsg = 'Camera error: ${e.description}');
    }
  }

  Future<void> _onRecordPressed() async {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized) return;

    if (_isRecording) {
      // ── STOP ──
      XFile? file;
      try {
        file = await ctrl.stopVideoRecording();
      } on CameraException catch (e) {
        _showError('Could not stop recording: ${e.description}');
        return;
      }

      final sensorTimeline = bleService.stopCapture();

      if (!mounted) return;
      setState(() => _isRecording = false);

      Navigator.of(context).pop(
        RecordResult(
          videoPath: file.path,
          sensorTimeline: sensorTimeline,
        ),
      );
    } else {
      // ── START ──
      try {
        await ctrl.prepareForVideoRecording();
        await ctrl.startVideoRecording();
      } on CameraException catch (e) {
        _showError('Could not start recording: ${e.description}');
        return;
      }

      final t0 = DateTime.now().millisecondsSinceEpoch;
      if (bleService.isConnected.value) {
        bleService.startCapture(t0);
      }

      setState(() => _isRecording = true);
    }
  }

  void _showError(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(msg)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: _errorMsg != null
            ? _buildError()
            : !_cameraReady
                ? const Center(
                    child: CircularProgressIndicator(color: Colors.white54))
                : _buildCamera(),
      ),
    );
  }

  Widget _buildError() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline, color: Colors.red, size: 48),
            const SizedBox(height: 16),
            Text(
              _errorMsg!,
              style: const TextStyle(color: Colors.white70),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Go back'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCamera() {
    final ctrl = _controller!;

    return Stack(
      fit: StackFit.expand,
      children: [
        // ── Camera preview ───────────────────────────────────────────────
        Center(
          child: AspectRatio(
            aspectRatio: ctrl.value.aspectRatio,
            child: CameraPreview(ctrl),
          ),
        ),

        // ── Top bar: back button + BLE status ───────────────────────────
        Positioned(
          top: 12,
          left: 12,
          right: 12,
          child: Row(
            children: [
              // Back (only when not recording)
              if (!_isRecording)
                GestureDetector(
                  onTap: () => Navigator.of(context).pop(),
                  child: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.black54,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Icon(Icons.arrow_back,
                        color: Colors.white, size: 22),
                  ),
                ),
              const Spacer(),
              // BLE indicator
              ValueListenableBuilder<bool>(
                valueListenable: bleService.isConnected,
                builder: (_, connected, __) {
                  return Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.black54,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: 8,
                          height: 8,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: connected ? Colors.greenAccent : Colors.grey,
                          ),
                        ),
                        const SizedBox(width: 6),
                        Text(
                          connected ? 'Sensor connected' : 'No sensor',
                          style: const TextStyle(
                              color: Colors.white70, fontSize: 12),
                        ),
                      ],
                    ),
                  );
                },
              ),
            ],
          ),
        ),

        // ── Record button ────────────────────────────────────────────────
        Positioned(
          bottom: 40,
          left: 0,
          right: 0,
          child: Center(
            child: GestureDetector(
              onTap: _onRecordPressed,
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                width: 72,
                height: 72,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white, width: 4),
                  color: _isRecording ? Colors.red : Colors.white24,
                ),
                child: Center(
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    width: _isRecording ? 24 : 48,
                    height: _isRecording ? 24 : 48,
                    decoration: BoxDecoration(
                      color: Colors.red,
                      borderRadius: BorderRadius.circular(
                          _isRecording ? 4 : 24),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),

        // ── Recording indicator ──────────────────────────────────────────
        if (_isRecording)
          Positioned(
            top: 16,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                    color: Colors.red.withValues(alpha: 0.85),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.fiber_manual_record,
                        color: Colors.white, size: 12),
                    SizedBox(width: 6),
                    Text('REC',
                        style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 12)),
                  ],
                ),
              ),
            ),
          ),
      ],
    );
  }
}
