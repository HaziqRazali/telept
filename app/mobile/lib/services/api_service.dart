import 'dart:convert';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:dio/dio.dart';

import '../config.dart';
import '../models/mesh_frame.dart';
import '../models/mesh_meta.dart';
import '../models/mhr_params_result.dart';
import 'obj_parser.dart';

/// Result of processing a video on the server (OBJ path).
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
          connectTimeout: const Duration(seconds: 10),
          receiveTimeout: AppConfig.uploadTimeout,
          sendTimeout: AppConfig.uploadTimeout,
          responseType: ResponseType.bytes,
        )) {
    // Attach the server API key on every request when configured.
    _dio.interceptors.add(InterceptorsWrapper(
      onRequest: (options, handler) {
        final key = AppConfig.serverApiKey;
        if (key.isNotEmpty) {
          options.headers['X-Api-Key'] = key;
        }
        handler.next(options);
      },
    ));
  }

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

  // ---------------------------------------------------------------------------
  // Option B: compact SMPL binary  (fast path)
  // ---------------------------------------------------------------------------

  /// Upload [videoPath], get MHR params binary back, run FK on device.
  ///
  /// [onProgress] 0.0–0.5 upload, 0.5–0.99 server inference, 1.0 done.
  Future<MhrParamsResult> processVideoParams(
    String videoPath, {
    void Function(double progress)? onProgress,
  }) async {
    // 1. Upload
    final formData = FormData.fromMap({
      'video': await MultipartFile.fromFile(videoPath, filename: 'recording.mp4'),
    });

    final uploadResponse = await _dio.post(
      AppConfig.processParamsEndpoint,
      data: formData,
      options: Options(responseType: ResponseType.plain),
      onSendProgress: (sent, total) {
        if (total > 0) onProgress?.call((sent / total) * 0.5);
      },
    );

    if (uploadResponse.statusCode != 200) {
      throw ApiException(
        'Server returned status ${uploadResponse.statusCode}',
        statusCode: uploadResponse.statusCode,
      );
    }

    final jobId =
        (json.decode(uploadResponse.data as String) as Map<String, dynamic>)['job_id'] as String;

    // 2. Poll
    while (true) {
      await Future.delayed(const Duration(milliseconds: 600));
      final progResponse = await _dio.get(
        AppConfig.progressEndpoint(jobId),
        options: Options(responseType: ResponseType.plain),
      );
      final body = json.decode(progResponse.data as String) as Map<String, dynamic>;
      final status = body['status'] as String;

      if (status == 'error') {
        throw ApiException(body['error'] as String? ?? 'Server processing failed');
      }
      if (status == 'processing') {
        final done  = (body['frames_done']  as num).toInt();
        final total = (body['total_frames'] as num).toInt();
        if (total > 0) {
          onProgress?.call(0.5 + (done / total) * 0.49);
        } else {
          onProgress?.call(0.5);
        }
      }
      if (status == 'done') {
        onProgress?.call(0.99);
        break;
      }
    }

    // 3. Fetch binary
    final resultResponse = await _dio.get(
      AppConfig.resultParamsEndpoint(jobId),
      options: Options(responseType: ResponseType.bytes),
    );
    if (resultResponse.statusCode != 200) {
      throw ApiException('Server returned ${resultResponse.statusCode}',
          statusCode: resultResponse.statusCode);
    }

    final binBytes = Uint8List.fromList(resultResponse.data as List<int>);
    onProgress?.call(1.0);
    return MhrParamsResult.fromBinary(binBytes);
  }

  // ---------------------------------------------------------------------------
  // Streaming SMPL params path
  // ---------------------------------------------------------------------------

  /// Upload [videoPath] and return the job_id immediately (does not wait for processing).
  ///
  /// [onSendProgress] reports upload fraction 0.0–1.0.
  Future<String> uploadVideo(
    String videoPath, {
    void Function(double progress)? onSendProgress,
  }) async {
    final formData = FormData.fromMap({
      'video': await MultipartFile.fromFile(videoPath, filename: 'recording.mp4'),
    });

    final response = await _dio.post(
      AppConfig.processParamsEndpoint,
      data: formData,
      options: Options(responseType: ResponseType.plain),
      onSendProgress: (sent, total) {
        if (total > 0) onSendProgress?.call(sent / total);
      },
    );

    if (response.statusCode != 200) {
      throw ApiException(
        'Server returned status ${response.statusCode}',
        statusCode: response.statusCode,
      );
    }

    return (json.decode(response.data as String) as Map<String, dynamic>)['job_id'] as String;
  }

  /// Poll /progress/{jobId} once and return the status map.
  Future<Map<String, dynamic>> pollProgress(String jobId) async {
    final response = await _dio.get(
      AppConfig.progressEndpoint(jobId),
      options: Options(responseType: ResponseType.plain),
    );
    return json.decode(response.data as String) as Map<String, dynamic>;
  }

  /// Fetch MHR params starting from [fromFrame] (frames ready so far).
  ///
  /// Returns a [MhrParamsResult] with however many frames are available.
  /// Safe to call while the job is still processing.
  Future<MhrParamsResult> fetchPartialParams(String jobId, int fromFrame) async {
    final response = await _dio.get(
      AppConfig.resultParamsPartialEndpoint(jobId, fromFrame),
      options: Options(responseType: ResponseType.bytes),
    );
    if (response.statusCode != 200) {
      throw ApiException(
        'Server returned ${response.statusCode}',
        statusCode: response.statusCode,
      );
    }
    final bytes = Uint8List.fromList(response.data as List<int>);
    return MhrParamsResult.fromBinary(bytes);
  }

  // ---------------------------------------------------------------------------
  // Legacy OBJ path (kept for fallback / debugging)
  // ---------------------------------------------------------------------------

  /// Upload [videoPath] to the server and return parsed mesh frames (OBJ path).
  Future<ProcessingResult> processVideo(
    String videoPath, {
    void Function(double progress)? onProgress,
  }) async {
    final formData = FormData.fromMap({
      'video': await MultipartFile.fromFile(videoPath, filename: 'recording.mp4'),
    });

    final uploadResponse = await _dio.post(
      AppConfig.processEndpoint,
      data: formData,
      options: Options(responseType: ResponseType.plain),
      onSendProgress: (sent, total) {
        if (total > 0 && onProgress != null) {
          onProgress((sent / total) * 0.5);
        }
      },
    );

    if (uploadResponse.statusCode != 200) {
      throw ApiException(
        'Server returned status ${uploadResponse.statusCode}',
        statusCode: uploadResponse.statusCode,
      );
    }

    final uploadBody = json.decode(uploadResponse.data as String) as Map<String, dynamic>;
    final jobId = uploadBody['job_id'] as String;

    while (true) {
      await Future.delayed(const Duration(milliseconds: 600));
      final progResponse = await _dio.get(
        AppConfig.progressEndpoint(jobId),
        options: Options(responseType: ResponseType.plain),
      );
      final body = json.decode(progResponse.data as String) as Map<String, dynamic>;
      final status = body['status'] as String;

      if (status == 'error') {
        throw ApiException(body['error'] as String? ?? 'Server processing failed');
      }
      if (status == 'processing') {
        final done  = (body['frames_done']  as num).toInt();
        final total = (body['total_frames'] as num).toInt();
        if (total > 0) {
          onProgress?.call(0.5 + (done / total) * 0.49);
        } else {
          onProgress?.call(0.5);
        }
      }
      if (status == 'done') {
        onProgress?.call(0.99);
        break;
      }
    }

    final resultResponse = await _dio.get(
      AppConfig.resultEndpoint(jobId),
      options: Options(responseType: ResponseType.bytes),
    );
    if (resultResponse.statusCode != 200) {
      throw ApiException('Server returned ${resultResponse.statusCode}',
          statusCode: resultResponse.statusCode);
    }

    final zipBytes = Uint8List.fromList(resultResponse.data as List<int>);
    onProgress?.call(1.0);
    return _parseZipResponse(zipBytes);
  }

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
