import 'dart:io';

import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../services/api_service.dart';
import '../widgets/mesh_viewer.dart';

/// Full-screen viewer: video on the left, 3D mesh on the right.
/// Single slim control bar at the bottom — maximum viewing space.
class ViewerScreen extends StatefulWidget {
  final ProcessingResult result;
  final String videoPath;

  const ViewerScreen({
    super.key,
    required this.result,
    required this.videoPath,
  });

  @override
  State<ViewerScreen> createState() => _ViewerScreenState();
}

class _ViewerScreenState extends State<ViewerScreen>
    with SingleTickerProviderStateMixin {
  int _currentFrame = 0;
  bool _isPlaying = false;
  late AnimationController _playController;

  late VideoPlayerController _videoController;
  bool _videoInitialized = false;

  int get _totalFrames => widget.result.frames.length;
  double get _fps => widget.result.meta.fps;

  @override
  void initState() {
    super.initState();

    final totalDurationMs =
        _totalFrames > 0 ? (_totalFrames / _fps * 1000).round() : 1000;
    _playController = AnimationController(
      vsync: this,
      duration: Duration(milliseconds: totalDurationMs),
    )..addListener(_onPlayTick);

    _videoController = VideoPlayerController.file(File(widget.videoPath))
      ..initialize().then((_) {
        if (mounted) setState(() => _videoInitialized = true);
      });
  }

  void _onPlayTick() {
    if (!_isPlaying) return;
    final frame = (_playController.value * (_totalFrames - 1)).round();
    if (frame != _currentFrame) {
      setState(() => _currentFrame = frame.clamp(0, _totalFrames - 1));
    }
    // Keep video in sync with mesh timeline
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
    _playController.dispose();
    _videoController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final frame = widget.result.frames[_currentFrame];

    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            // ── Split view ──────────────────────────────────────────────
            Expanded(
              child: Row(
                children: [
                  // Left: video playback
                  Expanded(
                    child: _videoInitialized
                        ? Center(
                            child: AspectRatio(
                              aspectRatio: _videoController.value.aspectRatio,
                              child: VideoPlayer(_videoController),
                            ),
                          )
                        : const Center(
                            child: CircularProgressIndicator(
                              color: Colors.white54,
                            ),
                          ),
                  ),

                  // Thin divider
                  Container(width: 1, color: Colors.white12),

                  // Right: 3D mesh viewer
                  Expanded(
                    child: MeshViewer(frame: frame),
                  ),
                ],
              ),
            ),

            // ── Bottom controls ─────────────────────────────────────────
            Container(
              color: Colors.black87,
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
              child: Row(
                children: [
                  // Back
                  IconButton(
                    icon: const Icon(Icons.arrow_back,
                        color: Colors.white70, size: 22),
                    onPressed: () => Navigator.pop(context),
                    tooltip: 'Back',
                  ),

                  // Play / Pause
                  IconButton(
                    icon: Icon(
                      _isPlaying ? Icons.pause : Icons.play_arrow,
                      color: Colors.white,
                      size: 26,
                    ),
                    onPressed: _totalFrames > 1 ? _togglePlayback : null,
                  ),

                  // Scrub slider
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
                        value: _currentFrame.toDouble(),
                        min: 0,
                        max: (_totalFrames - 1).toDouble(),
                        divisions: _totalFrames > 1 ? _totalFrames - 1 : 1,
                        onChanged: (v) => _seekToFrame(v.round()),
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

