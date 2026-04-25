import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../config.dart';
import '../models/mesh_frame.dart';
import '../models/smpl_params_result.dart';
import '../services/api_service.dart';
import '../services/smpl_model.dart';
import 'viewer_screen.dart';

/// Home screen with a "Record Video" button and server upload flow.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _api = ApiService();
  final _picker = ImagePicker();

  bool _isProcessing = false;
  String _statusText = '';
  double _uploadProgress = 0;
  double _serverProgress = 0; // 0.0–1.0, frames_done/total_frames

  Future<void> _recordAndProcess() async {
    final XFile? video = await _picker.pickVideo(
      source: ImageSource.camera,
      maxDuration: AppConfig.maxVideoDurationSec > 0
          ? Duration(seconds: AppConfig.maxVideoDurationSec)
          : null,
    );
    if (video == null) {
      _showSnack('Recording cancelled');
      return;
    }
    await _processVideo(video.path);
  }

  Future<void> _pickVideoFromGallery() async {
    final XFile? video = await _picker.pickVideo(source: ImageSource.gallery);
    if (video == null) {
      _showSnack('No video selected');
      return;
    }
    await _processVideo(video.path);
  }

  // ---------------------------------------------------------------------------
  // Core streaming processing flow
  // ---------------------------------------------------------------------------

  Future<void> _processVideo(String videoPath) async {
    setState(() {
      _isProcessing = true;
      _statusText = 'Checking server...';
      _uploadProgress = 0;
      _serverProgress = 0;
    });

    try {
      // 1. Health check
      final reachable = await _api.isServerReachable();
      if (!reachable) {
        throw const ApiException('Cannot reach server. Check Wi-Fi and server URL.');
      }

      // 2. Upload video, get job_id immediately
      setState(() => _statusText = 'Uploading video...');
      final jobId = await _api.uploadVideo(
        videoPath,
        onSendProgress: (p) => setState(() {
          _uploadProgress = p;
          _statusText = 'Uploading video... ${(p * 100).toInt()}%';
        }),
      );

      // 3. Poll until minStartFrames are ready (or job is done)
      setState(() => _statusText = 'Processing on server...');
      final totalFramesNotifier = ValueNotifier<int>(0);
      while (mounted) {
        await Future.delayed(const Duration(milliseconds: 800));
        if (!mounted) return;

        final progress = await _api.pollProgress(jobId);
        final status = progress['status'] as String;

        if (status == 'error') {
          throw ApiException(progress['error'] as String? ?? 'Server processing failed');
        }

        final framesDone = (progress['frames_done'] as num).toInt();
        final totalFrames = (progress['total_frames'] as num).toInt();
        if (totalFrames > 0) {
          totalFramesNotifier.value = totalFrames;
          setState(() {
            _serverProgress = framesDone / totalFrames;
            _statusText = 'Processing on server... ${(_serverProgress * 100).toInt()}%';
          });
        }

        if (status == 'done' || framesDone >= AppConfig.minStartFrames) break;
      }
      if (!mounted) return;

      // 4. Fetch the initial batch of frames
      final initialParams = await _api.fetchPartialParams(jobId, 0);
      if (!mounted) return;

      // 5. Run FK on device for the initial batch
      setState(() => _statusText = 'Computing 3D frames...');
      final smpl = SmplModel.instance;
      final initialFrames = await _runFk(smpl, initialParams);
      if (!mounted) return;

      final fps = initialParams.fps > 0 ? initialParams.fps : 30.0;
      final framesNotifier = ValueNotifier<List<MeshFrame>>(initialFrames);

      setState(() {
        _statusText = '';
        // Keep _isProcessing = true so buttons remain disabled until background loop ends.
      });

      // 6. Open the viewer immediately with what we have
      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ViewerScreen.streaming(
            framesNotifier: framesNotifier,
            streamingFps: fps,
            videoPath: videoPath,
            totalFramesNotifier: totalFramesNotifier,
            focalLength: initialParams.focalLength,
          ),
        ),
      );

      // 7. Continue fetching remaining frames in the background
      await _streamRemainingFrames(
        jobId: jobId,
        notifier: framesNotifier,
        totalFramesNotifier: totalFramesNotifier,
        smpl: smpl,
        nextFrame: initialParams.frameCount,
      );
    } on ApiException catch (e) {
      _showError(e.message);
    } catch (e) {
      _showError('Unexpected error: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isProcessing = false;
          _statusText = '';
          _serverProgress = 0;
        });
      }
    }
  }

  /// Run SMPL forward kinematics on [params] and return the resulting frames.
  Future<List<MeshFrame>> _runFk(SmplModel smpl, SmplParamsResult params) async {
    final frames = <MeshFrame>[];
    for (int i = 0; i < params.frameCount; i++) {
      if (params.isValid(i)) {
        frames.add(smpl.forward(
          go: params.go(i),
          bodyPose: params.bodyPose(i),
          betas: params.betas(i),
          camT: params.camT(i),
        ));
      } else {
        frames.add(smpl.forward(
          go: List.filled(3, 0.0),
          bodyPose: List.filled(63, 0.0),
          betas: List.filled(10, 0.0),
        ));
      }
      if (i % 10 == 0) await Future.microtask(() {});
    }
    return frames;
  }

  /// Background loop that keeps appending frames to [notifier] until the job is done.
  Future<void> _streamRemainingFrames({
    required String jobId,
    required ValueNotifier<List<MeshFrame>> notifier,
    required ValueNotifier<int> totalFramesNotifier,
    required SmplModel smpl,
    required int nextFrame,
  }) async {
    try {
      while (mounted) {
        await Future.delayed(const Duration(seconds: 2));
        if (!mounted) break;

        final progress = await _api.pollProgress(jobId);
        final status = progress['status'] as String;

        if (status == 'error') break; // non-fatal; user already has partial data

        final framesDone  = (progress['frames_done']  as num).toInt();
        final totalFrames = (progress['total_frames'] as num).toInt();

        // Keep totalFramesNotifier current so the scrubber grey bar stays accurate.
        if (totalFrames > 0) totalFramesNotifier.value = totalFrames;

        if (framesDone > nextFrame) {
          final partial = await _api.fetchPartialParams(jobId, nextFrame);
          if (partial.frameCount > 0) {
            final newFrames = await _runFk(smpl, partial);
            notifier.value = List.unmodifiable([...notifier.value, ...newFrames]);
            nextFrame += partial.frameCount;
          }
        }

        if (status == 'done') break;
      }
    } catch (_) {
      // Background fetch errors are non-fatal; the user can still scrub what they have.
    }
  }

  void _showSnack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  void _showError(String msg) {
    setState(() {
      _isProcessing = false;
      _statusText = '';
    });
    if (!mounted) return;
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Error'),
        content: Text(msg),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  Future<void> _showServerSettings(BuildContext context) async {
    final serverController = TextEditingController(text: AppConfig.serverUrl);
    final geminiController = TextEditingController(text: AppConfig.geminiApiKey);
    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Settings'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'GPU Server URL',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 13),
            ),
            const SizedBox(height: 4),
            const Text(
              'Both devices must be on the same Wi-Fi.',
              style: TextStyle(fontSize: 12, color: Colors.grey),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: serverController,
              autocorrect: false,
              keyboardType: TextInputType.url,
              decoration: const InputDecoration(
                labelText: 'Server URL',
                hintText: 'http://192.168.x.x:8000',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 20),
            const Text(
              'Gemini API Key',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 13),
            ),
            const SizedBox(height: 4),
            const Text(
              'Get a free key at aistudio.google.com',
              style: TextStyle(fontSize: 12, color: Colors.grey),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: geminiController,
              autocorrect: false,
              obscureText: true,
              decoration: const InputDecoration(
                labelText: 'Gemini API Key',
                hintText: 'AIza...',
                border: OutlineInputBorder(),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () async {
              await AppConfig.setServerUrl(serverController.text.trim());
              await AppConfig.setGeminiApiKey(geminiController.text.trim());
              if (ctx.mounted) Navigator.pop(ctx);
              _showSnack('Settings saved');
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
    // Controllers are intentionally not disposed here: showDialog resolves
    // when pop() is called, but the close animation is still running at that
    // point. Disposing now would crash any TextField rebuild during the
    // animation. They will be GC-collected once they go out of scope.
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('TelePT Body Capture'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            tooltip: 'Server settings',
            onPressed: () => _showServerSettings(context),
          ),
        ],
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Logo / Icon
              Icon(
                Icons.accessibility_new,
                size: 100,
                color: Theme.of(context).colorScheme.primary,
              ),
              const SizedBox(height: 24),
              Text(
                'Capture a short video to reconstruct\na 3D body mesh.',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyLarge,
              ),
              const SizedBox(height: 48),

              // Record button
              FilledButton.icon(
                onPressed: _isProcessing ? null : _recordAndProcess,
                icon: const Icon(Icons.videocam),
                label: const Text('Record Video'),
                style: FilledButton.styleFrom(
                  minimumSize: const Size(240, 56),
                  textStyle: const TextStyle(fontSize: 18),
                ),
              ),
              const SizedBox(height: 16),

              // Pick from gallery
              OutlinedButton.icon(
                onPressed: _isProcessing ? null : _pickVideoFromGallery,
                icon: const Icon(Icons.photo_library),
                label: const Text('Pick from Gallery'),
                style: OutlinedButton.styleFrom(
                  minimumSize: const Size(240, 56),
                  textStyle: const TextStyle(fontSize: 18),
                ),
              ),
              const SizedBox(height: 32),

              // Progress indicator
              if (_isProcessing) ...[
                if (_uploadProgress < 0.5)
                  LinearProgressIndicator(value: _uploadProgress * 2)
                else if (_serverProgress > 0)
                  LinearProgressIndicator(value: _serverProgress)
                else
                  const LinearProgressIndicator(), // indeterminate until first frame reported
                const SizedBox(height: 12),
                Text(
                  _statusText,
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
