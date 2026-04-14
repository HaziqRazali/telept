/// TelePT 3D Body Capture – App configuration.
library;

import 'package:shared_preferences/shared_preferences.dart';

class AppConfig {
  AppConfig._();

  static const String _kServerUrl = 'server_url';

  /// Fallback URL compiled into the app. Edit this if you always use the same
  /// server and don't want to configure it on first launch.
  static const String defaultServerUrl = 'http://192.168.1.68:8000';

  // Runtime-mutable – changed via the in-app settings dialog.
  static String _serverUrl = defaultServerUrl;

  /// Current server base URL (stored in SharedPreferences).
  static String get serverUrl => _serverUrl;

  static String get processEndpoint => '$_serverUrl/process';
  static String get healthEndpoint => '$_serverUrl/health';

  /// Call once in main() before runApp.
  static Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _serverUrl = prefs.getString(_kServerUrl) ?? defaultServerUrl;
  }

  /// Persist a new server URL. Takes effect immediately.
  static Future<void> setServerUrl(String url) async {
    _serverUrl = url.trimRight().replaceAll(RegExp(r'/+$'), '');
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kServerUrl, _serverUrl);
  }

  /// Upload / processing timeout (large videos can take a while).
  static const Duration uploadTimeout = Duration(minutes: 10);

  /// Maximum video recording duration in seconds (0 = unlimited).
  static const int maxVideoDurationSec = 60;
}
