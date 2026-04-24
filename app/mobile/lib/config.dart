/// TelePT 3D Body Capture – App configuration.
library;

import 'package:shared_preferences/shared_preferences.dart';

class AppConfig {
  AppConfig._();

  static const String _kServerUrl = 'server_url';
  static const String _kGeminiApiKey = 'gemini_api_key';

  /// Fallback URL compiled into the app. Edit this if you always use the same
  /// server and don't want to configure it on first launch.
  static const String defaultServerUrl = 'http://192.168.1.68:8000';

  // Runtime-mutable – changed via the in-app settings dialog.
  static String _serverUrl = defaultServerUrl;
  static String _geminiApiKey = '';

  /// Current server base URL (stored in SharedPreferences).
  static String get serverUrl => _serverUrl;
  static String get geminiApiKey => _geminiApiKey;
  static bool get hasGeminiKey => _geminiApiKey.isNotEmpty;

  static String get processEndpoint => '$_serverUrl/process';
  static String get processParamsEndpoint => '$_serverUrl/process_params';
  static String get healthEndpoint => '$_serverUrl/health';
  static String progressEndpoint(String jobId) => '$_serverUrl/progress/$jobId';
  static String resultEndpoint(String jobId) => '$_serverUrl/result/$jobId';
  static String resultParamsEndpoint(String jobId) => '$_serverUrl/result_params/$jobId';
  static String resultParamsPartialEndpoint(String jobId, int fromFrame) =>
      '$_serverUrl/result_params_partial/$jobId?from_frame=$fromFrame';

  /// Minimum frames that must be ready before the viewer opens in streaming mode.
  static const int minStartFrames = 30;

  /// Call once in main() before runApp.
  static Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _serverUrl = prefs.getString(_kServerUrl) ?? defaultServerUrl;
    _geminiApiKey = prefs.getString(_kGeminiApiKey) ?? '';
  }

  /// Persist a new server URL. Takes effect immediately.
  static Future<void> setServerUrl(String url) async {
    _serverUrl = url.trimRight().replaceAll(RegExp(r'/+$'), '');
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kServerUrl, _serverUrl);
  }

  /// Persist a new Gemini API key. Takes effect immediately.
  static Future<void> setGeminiApiKey(String key) async {
    _geminiApiKey = key.trim();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kGeminiApiKey, _geminiApiKey);
  }

  /// Upload / processing timeout (large videos can take a while).
  static const Duration uploadTimeout = Duration(minutes: 10);

  /// Maximum video recording duration in seconds (0 = unlimited).
  static const int maxVideoDurationSec = 60;
}
