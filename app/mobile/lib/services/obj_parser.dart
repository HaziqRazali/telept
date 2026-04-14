import 'dart:math' as math;
import '../models/mesh_frame.dart';

/// Parses a Wavefront .OBJ string into a [MeshFrame].
///
/// Supports:
///  - `v x y z`     (vertex positions)
///  - `vn x y z`    (vertex normals)
///  - `f v1 v2 v3`  (triangle faces, 1-indexed, optionally `v/vt/vn`)
///
/// Only triangular faces are supported. Quads are split into 2 triangles.
class ObjParser {
  ObjParser._();

  static MeshFrame parse(String objText) {
    final vertices = <double>[];
    final normals = <double>[];
    final indices = <int>[];

    final lines = objText.split('\n');

    for (final line in lines) {
      final trimmed = line.trim();
      if (trimmed.isEmpty || trimmed.startsWith('#')) continue;

      final parts = trimmed.split(RegExp(r'\s+'));
      final keyword = parts[0];

      if (keyword == 'v' && parts.length >= 4) {
        vertices.add(double.tryParse(parts[1]) ?? 0);
        vertices.add(double.tryParse(parts[2]) ?? 0);
        vertices.add(double.tryParse(parts[3]) ?? 0);
      } else if (keyword == 'vn' && parts.length >= 4) {
        normals.add(double.tryParse(parts[1]) ?? 0);
        normals.add(double.tryParse(parts[2]) ?? 0);
        normals.add(double.tryParse(parts[3]) ?? 0);
      } else if (keyword == 'f') {
        // Parse face vertex indices (handles v, v/vt, v/vt/vn, v//vn)
        final faceIndices = <int>[];
        for (int i = 1; i < parts.length; i++) {
          final vertexStr = parts[i].split('/')[0];
          final idx = int.tryParse(vertexStr);
          if (idx != null) {
            faceIndices.add(idx - 1); // OBJ is 1-indexed
          }
        }

        // Triangulate (fan triangulation for convex polygons)
        for (int i = 1; i < faceIndices.length - 1; i++) {
          indices.add(faceIndices[0]);
          indices.add(faceIndices[i]);
          indices.add(faceIndices[i + 1]);
        }
      }
    }

    // If no normals provided, compute per-vertex normals from faces
    final List<double> finalNormals;
    if (normals.isEmpty && vertices.isNotEmpty && indices.isNotEmpty) {
      finalNormals = _computeNormals(vertices, indices);
    } else {
      finalNormals = normals;
    }

    return MeshFrame(
      vertices: vertices,
      indices: indices,
      normals: finalNormals,
    );
  }

  /// Compute smooth per-vertex normals by averaging face normals.
  static List<double> _computeNormals(List<double> verts, List<int> idxs) {
    final n = verts.length ~/ 3;
    final normals = List<double>.filled(n * 3, 0);

    for (int i = 0; i < idxs.length; i += 3) {
      final i0 = idxs[i], i1 = idxs[i + 1], i2 = idxs[i + 2];

      final ax = verts[i1 * 3] - verts[i0 * 3];
      final ay = verts[i1 * 3 + 1] - verts[i0 * 3 + 1];
      final az = verts[i1 * 3 + 2] - verts[i0 * 3 + 2];

      final bx = verts[i2 * 3] - verts[i0 * 3];
      final by = verts[i2 * 3 + 1] - verts[i0 * 3 + 1];
      final bz = verts[i2 * 3 + 2] - verts[i0 * 3 + 2];

      // Cross product a × b
      final nx = ay * bz - az * by;
      final ny = az * bx - ax * bz;
      final nz = ax * by - ay * bx;

      for (final vi in [i0, i1, i2]) {
        normals[vi * 3] += nx;
        normals[vi * 3 + 1] += ny;
        normals[vi * 3 + 2] += nz;
      }
    }

    // Normalize
    for (int i = 0; i < n; i++) {
      final x = normals[i * 3];
      final y = normals[i * 3 + 1];
      final z = normals[i * 3 + 2];
      final len = math.sqrt(x * x + y * y + z * z);
      if (len > 1e-8) {
        normals[i * 3] /= len;
        normals[i * 3 + 1] /= len;
        normals[i * 3 + 2] /= len;
      }
    }

    return normals;
  }
}
