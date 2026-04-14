import 'dart:convert';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:dio/dio.dart';

import '../config.dart';
import '../models/mesh_frame.dart';
import '../models/mesh_meta.dart';
import 'obj_parser.dart';

/// Result of processing a video on the server.
class ProcessingResult {
  final MeshMeta meta;
  final List<MeshFrame> frames;

  const ProcessingResult({required this.meta, required this.frames});
}

/// Service that uploads a video to the server and returns parsed mesh frames.
class ApiService {
  final Dio _dio;

  ApiService()
      : _dio = Dio(BaseOptions(
          // No fixed baseUrl — built dynamically so URL changes take effect
          // immediately after the user updates it in Settings.
          connectTimeout: const Duration(seconds: 10),
          receiveTimeout: AppConfig.uploadTimeout,
          sendTimeout: AppConfig.uploadTimeout,
          responseType: ResponseType.bytes,
        ));

  /// Check server health.
  Future<bool> isServerReachable() async {
    try {
      final resp = await _dio.get(
        AppConfig.healthEndpoint,
        options: Options(responseType: ResponseType.json),
      );
      return resp.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Upload [videoPath] to the server and return parsed mesh frames.
  ///
  /// [onProgress] reports upload progress as a value between 0.0 and 1.0.
  Future<ProcessingResult> processVideo(
    String videoPath, {
    void Function(double progress)? onProgress,
  }) async {
    final formData = FormData.fromMap({
      'video': await MultipartFile.fromFile(videoPath, filename: 'recording.mp4'),
    });

    final response = await _dio.post(
      AppConfig.processEndpoint,
      data: formData,
      onSendProgress: (sent, total) {
        if (total > 0 && onProgress != null) {
          onProgress(sent / total);
        }
      },
    );

    if (response.statusCode != 200) {
      throw ApiException(
        'Server returned status ${response.statusCode}',
        statusCode: response.statusCode,
      );
    }

    // Response body is a ZIP file in bytes
    final Uint8List zipBytes = Uint8List.fromList(response.data as List<int>);
    return _parseZipResponse(zipBytes);
  }

  /// Parse the ZIP response into [ProcessingResult].
  ProcessingResult _parseZipResponse(Uint8List zipBytes) {
    final archive = ZipDecoder().decodeBytes(zipBytes);

    MeshMeta? meta;
    final objEntries = <String, ArchiveFile>{};

    for (final file in archive) {
      if (file.isFile) {
        final name = file.name;
        if (name == 'meta.json') {
          final jsonStr = utf8.decode(file.content as List<int>);
          meta = MeshMeta.fromJson(json.decode(jsonStr) as Map<String, dynamic>);
        } else if (name.endsWith('.obj')) {
          objEntries[name] = file;
        }
      }
    }

    if (meta == null) {
      throw const ApiException('meta.json not found in server response');
    }

    // Sort OBJ files by name to get frame order
    final sortedNames = objEntries.keys.toList()..sort();

    final frames = <MeshFrame>[];
    for (final name in sortedNames) {
      final objText = utf8.decode(objEntries[name]!.content as List<int>);
      frames.add(ObjParser.parse(objText));
    }

    if (frames.isEmpty) {
      throw const ApiException('No OBJ files found in server response');
    }

    return ProcessingResult(meta: meta, frames: frames);
  }
}

/// Custom exception for API errors.
class ApiException implements Exception {
  final String message;
  final int? statusCode;

  const ApiException(this.message, {this.statusCode});

  @override
  String toString() => 'ApiException($statusCode): $message';
}
