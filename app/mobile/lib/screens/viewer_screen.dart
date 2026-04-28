import 'dart:io';

import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_charts/charts.dart';
import 'package:video_player/video_player.dart';

import '../models/mesh_frame.dart';
import '../models/sensor_sample.dart';
import '../services/api_service.dart';
import '../widgets/mesh_viewer.dart';
import '../widgets/chat_panel.dart';

/// Full-screen viewer with a left icon-rail sidebar to show / hide three panels:
/// (1) Video playback  (2) 3D mesh  (3) Gemini chatbot
class ViewerScreen extends StatefulWidget {
  /// Static result (all frames available up-front).
  final ProcessingResult? result;

  /// Streaming result — frames are appended to this notifier over time.
  final ValueNotifier<List<MeshFrame>>? framesNotifier;

  /// FPS for streaming mode (ignored when [result] is provided).
  final double? streamingFps;

  final String videoPath;

  /// Standard constructor: all frames are already computed.
  const ViewerScreen({
    super.key,
    required ProcessingResult this.result,
    required this.videoPath,
    this.sensorTimeline,
  })  : framesNotifier = null,
        streamingFps = null,
        totalFramesNotifier = null,
        focalLength = 0.0;

  /// Total expected frame count for the full video (used for the buffer track).
  /// Updated by the home screen's polling loop as total_frames becomes known.
  final ValueNotifier<int>? totalFramesNotifier;

  /// Focal length in pixels from the server (used for perspective projection
  /// in overlay mode).  0 when unknown (uses a FOV-based fallback).
  final double focalLength;

  /// Optional sensor timeline recorded during video capture.
  /// When non-null, a sensor chart panel is available in the viewer.
  final List<SensorSample>? sensorTimeline;

  /// Streaming constructor: frames arrive incrementally via [framesNotifier].
  const ViewerScreen.streaming({
    super.key,
    required ValueNotifier<List<MeshFrame>> this.framesNotifier,
    required double this.streamingFps,
    required this.videoPath,
    this.totalFramesNotifier,
    this.focalLength = 0.0,
    this.sensorTimeline,
  }) : result = null;

  @override
  State<ViewerScreen> createState() => _ViewerScreenState();
}

class _ViewerScreenState extends State<ViewerScreen>
    with SingleTickerProviderStateMixin {
  // ── Panel visibility ────────────────────────────────────────────────
  bool _showVideo = true;
  bool _showMesh = true;
  bool _showChat = false;
  bool _showSensor = false;
  bool _overlayMesh = false;

  bool get _hasSensor => (widget.sensorTimeline?.isNotEmpty ?? false);

  // Sensor is now a bottom panel, not a column — only count top-row panels.
  int get _visibleCount =>
      (_showVideo ? 1 : 0) + (_showMesh ? 1 : 0) + (_showChat ? 1 : 0);

  void _toggle(String panel) {
    setState(() {
      if (panel == 'overlay') {
        _overlayMesh = !_overlayMesh;
        // Overlay requires both video and mesh to be on.
        if (_overlayMesh) {
          _showVideo = true;
          _showMesh = true;
        }
        return;
      }
      final wouldHide = switch (panel) {
        'video' => _showVideo,
        'mesh' => _showMesh,
        'chat' => _showChat,
        'sensor' => _showSensor,
        _ => false,
      };
      // Don't allow hiding the last visible panel
      if (wouldHide && _visibleCount <= 1) return;
      switch (panel) {
        case 'video':
          _showVideo = !_showVideo;
          if (!_showVideo) _overlayMesh = false;
        case 'mesh':
          _showMesh = !_showMesh;
          if (!_showMesh) _overlayMesh = false;
        case 'chat':
          _showChat = !_showChat;
        case 'sensor':
          _showSensor = !_showSensor;
      }
    });
  }

  // ── Frame accessors ─────────────────────────────────────────────────
  List<MeshFrame> get _frames =>
      widget.framesNotifier?.value ?? widget.result!.frames.cast<MeshFrame>();

  int get _totalFrames => _frames.length;

  /// Total expected frames for the full video (0 = unknown).
  int get _totalExpectedFrames =>
      widget.totalFramesNotifier?.value ?? 0;

  double get _fps =>
      widget.streamingFps ?? widget.result!.meta.fps;

  // ── Playback ────────────────────────────────────────────────────────
  int _currentFrame = 0;
  bool _isPlaying = false;
  late AnimationController _playController;

  late VideoPlayerController _videoController;
  bool _videoInitialized = false;

  @override
  void initState() {
    super.initState();

    // Start with whatever frames are available; duration will grow as more arrive.
    final initialFrames = _totalFrames;
    final initialDurationMs =
        initialFrames > 0 ? (initialFrames / _fps * 1000).round() : 1000;
    _playController = AnimationController(
      vsync: this,
      duration: Duration(milliseconds: initialDurationMs),
    )..addListener(_onPlayTick);

    // Listen for new frames and total-frame updates in streaming mode.
    widget.framesNotifier?.addListener(_onFramesUpdated);
    widget.totalFramesNotifier?.addListener(_onFramesUpdated);

    _videoController = VideoPlayerController.file(File(widget.videoPath))
      ..initialize().then((_) {
        if (mounted) setState(() => _videoInitialized = true);
      });
  }

  /// Called when new frames are appended in streaming mode.
  void _onFramesUpdated() {
    if (!mounted) return;
    setState(() {
      // Extend the controller duration to match the new total so the scrubber
      // max grows automatically.  Don't interrupt a running animation.
      if (!_isPlaying) {
        final newDurationMs = (_totalFrames / _fps * 1000).round();
        _playController.duration = Duration(milliseconds: newDurationMs);
      }
    });
  }

  void _onPlayTick() {
    if (!_isPlaying) return;
    final frame = (_playController.value * (_totalFrames - 1)).round();
    if (frame != _currentFrame) {
      setState(() => _currentFrame = frame.clamp(0, _totalFrames - 1));
    }
    if (_videoInitialized) {
      final videoMs = (_playController.value *
              _videoController.value.duration.inMilliseconds)
          .round();
      _videoController.seekTo(Duration(milliseconds: videoMs));
    }
  }

  void _togglePlayback() {
    setState(() {
      _isPlaying = !_isPlaying;
      if (_isPlaying) {
        if (_videoInitialized) _videoController.play();
        if (_currentFrame >= _totalFrames - 1) {
          _playController.forward(from: 0);
        } else {
          _playController.forward(
              from: _currentFrame / (_totalFrames - 1).toDouble());
        }
      } else {
        if (_videoInitialized) _videoController.pause();
        _playController.stop();
      }
    });
  }

  void _seekToFrame(int frame) {
    if (_isPlaying) {
      _playController.stop();
      if (_videoInitialized) _videoController.pause();
    }
    setState(() {
      _isPlaying = false;
      _currentFrame = frame.clamp(0, _totalFrames - 1);
    });
    if (_videoInitialized && _totalFrames > 1) {
      final ratio = _currentFrame / (_totalFrames - 1);
      final videoMs =
          (ratio * _videoController.value.duration.inMilliseconds).round();
      _videoController.seekTo(Duration(milliseconds: videoMs));
    }
  }

  @override
  void dispose() {
    widget.framesNotifier?.removeListener(_onFramesUpdated);
    widget.totalFramesNotifier?.removeListener(_onFramesUpdated);
    _playController.dispose();
    _videoController.dispose();
    super.dispose();
  }

  // ── Scrubber with optional buffer-track overlay ─────────────────────
  Widget _buildScrubber(BuildContext context) {
    final loaded   = _totalFrames;
    final expected = _totalExpectedFrames;
    final isStreaming = widget.totalFramesNotifier != null && expected > loaded;

    // In streaming mode: max = total expected so the grey "unloaded" region
    // is visible.  onChanged clamps to loaded frames so the user cannot seek
    // past what is ready.
    final sliderMax   = isStreaming ? (expected - 1).toDouble()
                                    : (loaded > 1 ? (loaded - 1).toDouble() : 1.0);
    final sliderValue = loaded > 0 ? _currentFrame.clamp(0, loaded - 1).toDouble() : 0.0;
    final divisions   = isStreaming ? (expected - 1) : (loaded > 1 ? loaded - 1 : 1);

    return SliderTheme(
      data: SliderTheme.of(context).copyWith(
        trackHeight: 2,
        thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 6),
        overlayShape: const RoundSliderOverlayShape(overlayRadius: 12),
        // Played portion (left of thumb)
        activeTrackColor: Colors.white,
        // Buffered-but-not-played portion (thumb → secondaryTrackValue)
        secondaryActiveTrackColor: Colors.white38,
        // Not-yet-downloaded portion (secondaryTrackValue → max)
        inactiveTrackColor: isStreaming ? Colors.white12 : Colors.white30,
      ),
      child: Slider(
        value: sliderValue,
        min: 0,
        max: sliderMax,
        divisions: divisions,
        // Marks the right edge of the buffered region.
        secondaryTrackValue: isStreaming
            ? (loaded > 0 ? (loaded - 1).toDouble() : 0.0)
            : null,
        onChanged: loaded > 1
            ? (v) => _seekToFrame(v.clamp(0, loaded - 1).round())
            : null,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final frames = _frames;
    final frame = frames.isEmpty ? null : frames[_currentFrame.clamp(0, frames.length - 1)];

    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            // ── Main area ──────────────────────────────────────────────
            Expanded(
              child: Row(
                children: [
                  // ── Left icon rail ─────────────────────────────────
                  _SideRail(
                    showVideo: _showVideo,
                    showMesh: _showMesh,
                    showChat: _showChat,                    showSensor: _showSensor,
                    hasSensor: _hasSensor,                    overlayMesh: _overlayMesh,
                    onToggle: _toggle,
                  ),

                  Container(width: 1, color: Colors.white12),

                  // ── Panel area ─────────────────────────────────────
                  Expanded(
                    child: Column(
                      children: [
                        // ── Top row: video / mesh / chat ──────────────
                        Expanded(
                          child: Row(
                            children: [
                              if (_overlayMesh && _showVideo && _showMesh) ...[
                                // Overlay mode: mesh drawn semi-transparently over video.
                                // Both the video and the overlay live inside the same
                                // AspectRatio so the overlay pixel space matches the
                                // displayed video pixel space exactly (no letterbox offset).
                                Expanded(
                                  child: _videoInitialized
                                      ? Center(
                                          child: AspectRatio(
                                            aspectRatio:
                                                _videoController.value.aspectRatio,
                                            child: Stack(
                                              fit: StackFit.expand,
                                              children: [
                                                VideoPlayer(_videoController),
                                                if (frame != null)
                                                  LayoutBuilder(
                                                    builder: (ctx, constraints) {
                                                      // Scale focal length from original
                                                      // video pixels → display pixels so
                                                      // the projection stays correct.
                                                      final displayW =
                                                          constraints.maxWidth;
                                                      final videoW = _videoController
                                                          .value.size.width;
                                                      final scaledFl = videoW > 0
                                                          ? widget.focalLength *
                                                              (displayW / videoW)
                                                          : widget.focalLength;
                                                      return Opacity(
                                                        opacity: 0.65,
                                                        child: MeshViewer(
                                                          frame: frame,
                                                          overlay: true,
                                                          focalLength: scaledFl,
                                                        ),
                                                      );
                                                    },
                                                  ),
                                              ],
                                            ),
                                          ),
                                        )
                                      : const Center(
                                          child: CircularProgressIndicator(
                                              color: Colors.white54),
                                        ),
                                ),
                              ] else ...[
                                if (_showVideo) ...[
                                  Expanded(
                                    child: _videoInitialized
                                        ? Center(
                                            child: AspectRatio(
                                              aspectRatio:
                                                  _videoController.value.aspectRatio,
                                              child: VideoPlayer(_videoController),
                                            ),
                                          )
                                        : const Center(
                                            child: CircularProgressIndicator(
                                                color: Colors.white54),
                                          ),
                                  ),
                                  if (_showMesh || _showChat)
                                    Container(width: 1, color: Colors.white12),
                                ],
                                if (_showMesh) ...[
                                  Expanded(
                                    child: frame != null
                                        ? MeshViewer(frame: frame)
                                        : const Center(
                                            child: CircularProgressIndicator(
                                                color: Colors.white54),
                                          ),
                                  ),
                                  if (_showChat)
                                    Container(width: 1, color: Colors.white12),
                                ],
                              ],
                              if (_showChat)
                                const Expanded(child: ChatPanel()),
                            ],
                          ),
                        ),
                        // ── Bottom row: sensor panel (full width) ─────
                        if (_showSensor && _hasSensor) ...[
                          Container(height: 1, color: Colors.white12),
                          SizedBox(
                            height: 220,
                            child: _SensorPanel(
                              timeline: widget.sensorTimeline!,
                              currentFrame: _currentFrame,
                              fps: _fps,
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                ],
              ),
            ),

            // ── Bottom controls ────────────────────────────────────────
            Container(
              color: Colors.black87,
              padding:
                  const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back,
                        color: Colors.white70, size: 22),
                    onPressed: () => Navigator.pop(context),
                    tooltip: 'Back',
                  ),
                  IconButton(
                    icon: Icon(
                      _isPlaying ? Icons.pause : Icons.play_arrow,
                      color: Colors.white,
                      size: 26,
                    ),
                    onPressed: _totalFrames > 1 ? _togglePlayback : null,
                  ),
                  Expanded(
                    child: _buildScrubber(context),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar icon rail
// ─────────────────────────────────────────────────────────────────────────────

class _SideRail extends StatelessWidget {
  final bool showVideo;
  final bool showMesh;
  final bool showChat;
  final bool showSensor;
  final bool hasSensor;
  final bool overlayMesh;
  final void Function(String) onToggle;

  const _SideRail({
    required this.showVideo,
    required this.showMesh,
    required this.showChat,
    required this.showSensor,
    required this.hasSensor,
    required this.overlayMesh,
    required this.onToggle,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 48,
      color: Colors.grey[900],
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          _RailButton(
            icon: Icons.videocam,
            label: 'Video',
            active: showVideo,
            activeColor: Colors.blue[300]!,
            onTap: () => onToggle('video'),
          ),
          const SizedBox(height: 8),
          _RailButton(
            icon: Icons.view_in_ar,
            label: 'Mesh',
            active: showMesh,
            activeColor: Colors.green[300]!,
            onTap: () => onToggle('mesh'),
          ),
          const SizedBox(height: 8),
          _RailButton(
            icon: Icons.layers,
            label: 'Overlay',
            active: overlayMesh,
            activeColor: Colors.orange[300]!,
            onTap: () => onToggle('overlay'),
          ),
          const SizedBox(height: 8),
          _RailButton(
            icon: Icons.chat_bubble_outline,
            label: 'Chat',
            active: showChat,
            activeColor: Colors.purple[300]!,
            onTap: () => onToggle('chat'),
          ),
          if (hasSensor) ...[
            const SizedBox(height: 8),
            _RailButton(
              icon: Icons.show_chart,
              label: 'Sensor',
              active: showSensor,
              activeColor: Colors.tealAccent,
              onTap: () => onToggle('sensor'),
            ),
          ],
        ],
      ),
    );
  }
}

class _RailButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool active;
  final Color activeColor;
  final VoidCallback onTap;

  const _RailButton({
    required this.icon,
    required this.label,
    required this.active,
    required this.activeColor,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: label,
      preferBelow: false,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(8),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            color: active ? activeColor.withValues(alpha: 0.15) : Colors.transparent,
          ),
          child: Icon(
            icon,
            size: 22,
            color: active ? activeColor : Colors.white30,
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sensor chart panel
// ─────────────────────────────────────────────────────────────────────────────

class _SensorPanel extends StatelessWidget {
  final List<SensorSample> timeline;
  final int currentFrame;
  final double fps;

  const _SensorPanel({
    required this.timeline,
    required this.currentFrame,
    required this.fps,
  });

  double get _cursorSec => fps > 0 ? currentFrame / fps : 0.0;

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.grey[900],
      padding: const EdgeInsets.fromLTRB(8, 12, 8, 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(left: 4, bottom: 6),
            child: Text(
              'Sensor Data',
              style: TextStyle(
                  color: Colors.white70,
                  fontSize: 12,
                  fontWeight: FontWeight.bold),
            ),
          ),
          Expanded(
            child: SfCartesianChart(
              backgroundColor: Colors.transparent,
              plotAreaBorderWidth: 0,
              legend: const Legend(
                isVisible: true,
                textStyle: TextStyle(color: Colors.white54, fontSize: 10),
              ),
              primaryXAxis: NumericAxis(
                title: const AxisTitle(
                    text: 'Time (s)',
                    textStyle: TextStyle(color: Colors.white54, fontSize: 10)),
                labelStyle:
                    const TextStyle(color: Colors.white54, fontSize: 10),
                majorGridLines:
                    const MajorGridLines(width: 0.3, color: Colors.white12),
                axisLine: const AxisLine(color: Colors.white24),
                plotBands: [
                  PlotBand(
                    isVisible: true,
                    start: _cursorSec,
                    end: _cursorSec,
                    borderWidth: 2,
                    borderColor: Colors.white70,
                  ),
                ],
              ),
              primaryYAxis: const NumericAxis(
                name: 'Force',
                title: AxisTitle(
                    text: 'Force (kg)',
                    textStyle: TextStyle(color: Colors.tealAccent, fontSize: 10)),
                labelStyle:
                    TextStyle(color: Colors.tealAccent, fontSize: 10),
                majorGridLines:
                    MajorGridLines(width: 0.3, color: Colors.white12),
                axisLine: AxisLine(color: Colors.white24),
              ),
              axes: const [
                NumericAxis(
                  name: 'Direction',
                  opposedPosition: true,
                  title: AxisTitle(
                      text: 'Direction (°)',
                      textStyle:
                          TextStyle(color: Colors.orangeAccent, fontSize: 10)),
                  labelStyle:
                      TextStyle(color: Colors.orangeAccent, fontSize: 10),
                  majorGridLines: MajorGridLines(width: 0),
                  axisLine: AxisLine(color: Colors.white24),
                ),
              ],
              series: [
                LineSeries<SensorSample, double>(
                  name: 'Force',
                  yAxisName: 'Force',
                  dataSource: timeline,
                  xValueMapper: (s, _) => s.timestampMs / 1000.0,
                  yValueMapper: (s, _) => s.force,
                  color: Colors.tealAccent,
                  width: 1.5,
                  markerSettings: const MarkerSettings(isVisible: false),
                ),
                LineSeries<SensorSample, double>(
                  name: 'Direction',
                  yAxisName: 'Direction',
                  dataSource: timeline,
                  xValueMapper: (s, _) => s.timestampMs / 1000.0,
                  yValueMapper: (s, _) => s.direction,
                  color: Colors.orangeAccent,
                  width: 1.5,
                  markerSettings: const MarkerSettings(isVisible: false),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
