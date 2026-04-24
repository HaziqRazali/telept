import 'dart:io';

import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../models/mesh_frame.dart';
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
  })  : framesNotifier = null,
        streamingFps = null;

  /// Streaming constructor: frames arrive incrementally via [framesNotifier].
  const ViewerScreen.streaming({
    super.key,
    required ValueNotifier<List<MeshFrame>> this.framesNotifier,
    required double this.streamingFps,
    required this.videoPath,
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

  int get _visibleCount =>
      (_showVideo ? 1 : 0) + (_showMesh ? 1 : 0) + (_showChat ? 1 : 0);

  void _toggle(String panel) {
    setState(() {
      final wouldHide = switch (panel) {
        'video' => _showVideo,
        'mesh' => _showMesh,
        'chat' => _showChat,
        _ => false,
      };
      // Don't allow hiding the last visible panel
      if (wouldHide && _visibleCount <= 1) return;
      switch (panel) {
        case 'video':
          _showVideo = !_showVideo;
        case 'mesh':
          _showMesh = !_showMesh;
        case 'chat':
          _showChat = !_showChat;
      }
    });
  }

  // ── Frame accessors ─────────────────────────────────────────────────
  List<MeshFrame> get _frames =>
      widget.framesNotifier?.value ?? widget.result!.frames.cast<MeshFrame>();

  int get _totalFrames => _frames.length;

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

    // Listen for new frames in streaming mode.
    widget.framesNotifier?.addListener(_onFramesUpdated);

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
    _playController.dispose();
    _videoController.dispose();
    super.dispose();
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
                    showChat: _showChat,
                    onToggle: _toggle,
                  ),

                  Container(width: 1, color: Colors.white12),

                  // ── Panel area ─────────────────────────────────────
                  Expanded(
                    child: Row(
                      children: [
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
                        if (_showChat)
                          const Expanded(child: ChatPanel()),
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
                    child: SliderTheme(
                      data: SliderTheme.of(context).copyWith(
                        trackHeight: 2,
                        thumbShape: const RoundSliderThumbShape(
                            enabledThumbRadius: 6),
                        overlayShape: const RoundSliderOverlayShape(
                            overlayRadius: 12),
                      ),
                      child: Slider(
                        value: _totalFrames > 0
                            ? _currentFrame.clamp(0, _totalFrames - 1).toDouble()
                            : 0.0,
                        min: 0,
                        max: _totalFrames > 1 ? (_totalFrames - 1).toDouble() : 1.0,
                        divisions: _totalFrames > 1 ? _totalFrames - 1 : 1,
                        onChanged: _totalFrames > 1
                            ? (v) => _seekToFrame(v.round())
                            : null,
                      ),
                    ),
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
  final void Function(String) onToggle;

  const _SideRail({
    required this.showVideo,
    required this.showMesh,
    required this.showChat,
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
            icon: Icons.chat_bubble_outline,
            label: 'Chat',
            active: showChat,
            activeColor: Colors.purple[300]!,
            onTap: () => onToggle('chat'),
          ),
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
