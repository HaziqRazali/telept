import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../config.dart';
import '../services/api_service.dart';
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

  Future<void> _recordAndProcess() async {
    // 1. Record video using the device camera
    final XFile? video = await _picker.pickVideo(
      source: ImageSource.camera,
      maxDuration: const Duration(seconds: 60),
    );

    if (video == null) {
      _showSnack('Recording cancelled');
      return;
    }

    setState(() {
      _isProcessing = true;
      _statusText = 'Checking server...';
      _uploadProgress = 0;
    });

    try {
      // 2. Check server health
      final reachable = await _api.isServerReachable();
      if (!reachable) {
        throw const ApiException('Cannot reach server. Check Wi-Fi and server URL.');
      }

      // 3. Upload and process
      setState(() => _statusText = 'Uploading video...');

      final result = await _api.processVideo(
        video.path,
        onProgress: (p) {
          setState(() {
            _uploadProgress = p;
            if (p >= 1.0) {
              _statusText = 'Processing on server...';
            }
          });
        },
      );

      setState(() {
        _isProcessing = false;
        _statusText = '';
      });

      if (!mounted) return;

      // 4. Navigate to the 3D viewer
      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ViewerScreen(result: result, videoPath: video.path),
        ),
      );
    } on ApiException catch (e) {
      _showError(e.message);
    } catch (e) {
      _showError('Unexpected error: $e');
    }
  }

  Future<void> _pickVideoFromGallery() async {
    final XFile? video = await _picker.pickVideo(source: ImageSource.gallery);
    if (video == null) {
      _showSnack('No video selected');
      return;
    }

    setState(() {
      _isProcessing = true;
      _statusText = 'Checking server...';
      _uploadProgress = 0;
    });

    try {
      final reachable = await _api.isServerReachable();
      if (!reachable) {
        throw const ApiException('Cannot reach server. Check Wi-Fi and server URL.');
      }

      setState(() => _statusText = 'Uploading video...');

      final result = await _api.processVideo(
        video.path,
        onProgress: (p) {
          setState(() {
            _uploadProgress = p;
            if (p >= 1.0) {
              _statusText = 'Processing on server...';
            }
          });
        },
      );

      setState(() {
        _isProcessing = false;
        _statusText = '';
      });

      if (!mounted) return;

      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ViewerScreen(result: result, videoPath: video.path),
        ),
      );
    } on ApiException catch (e) {
      _showError(e.message);
    } catch (e) {
      _showError('Unexpected error: $e');
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
                if (_uploadProgress < 1.0 && _statusText.contains('Uploading'))
                  LinearProgressIndicator(value: _uploadProgress)
                else
                  const CircularProgressIndicator(),
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
