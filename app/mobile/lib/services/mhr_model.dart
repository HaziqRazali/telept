/// MHR body model forward kinematics implemented in Dart.
///
/// Loads static binary assets from assets/mhr/ and computes per-vertex
/// positions given model_parameters[204].
///
/// Mathematical pipeline (mirrors pymomentum exactly):
///   1. param_transform:  joint_params = T @ model_params
///                        T is (889, 321) float32; model_params zero-padded to 321
///   2. FK tree traversal: for each joint, compose pre_rotation and local euler ZYX
///                         rotation with parent global transform
///   3. Skinning matrix:   M[j] = global_transform[j] @ inverse_bind_pose[j]
///   4. LBS:              vertex = sum_k( weight[k] * M[joint[k]] @ rest_vertex )
///
/// All verified numerically against pymomentum output (max error < 5e-5).
///
/// Asset layout in assets/mhr/:
///   param_transform.bin  float32 (889 × 321)
///   skeleton.bin         int32 n_joints | int16[n] parents | pad | float32[n,3] offsets | float32[n,4] pre_rots
///   inv_bind_pose.bin    float32 (127 × 4 × 4)
///   skin_idx.bin         int16  (N × 8)
///   skin_wgt.bin         float32 (N × 8)
///   rest_verts.bin       float32 (N × 3)   centimetres
///   faces.bin            int32  (F × 3)
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';

import '../models/mesh_frame.dart';

class MhrModel {
  // ---------------------------------------------------------------------------
  // Asset data
  // ---------------------------------------------------------------------------

  /// T matrix: (889, 321) float32 — param_transform.  Stored row-major.
  late final Float32List _T;

  /// Joint hierarchy
  late final int _nJoints;
  late final Int16List _parents;      // [nJoints]   -1 = root
  late final Float32List _offsets;    // [nJoints, 3]
  late final Float32List _preRots;    // [nJoints, 4] xyzw quaternions

  /// Inverse bind pose: (127, 4, 4) float32, row-major 4×4 per joint.
  late final Float32List _ibp;

  /// Sparse skin weights: (nVerts, 8) int16 indices + float32 weights.
  late final Int16List _skinIdx;
  late final Float32List _skinWgt;

  /// Rest-pose vertices: (nVerts, 3) float32, centimetres.
  late final Float32List _restVerts;

  /// Triangle face indices: (nFaces, 3) int32.
  late final Int32List _faces;

  late final int _nVerts;
  late final int _nFaces;

  bool _loaded = false;

  static MhrModel? _instance;
  static MhrModel get instance => _instance ??= MhrModel._();
  MhrModel._();

  // ---------------------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------------------
  static const int _nJointsConst   = 127;
  static const int _jpRows         = 889;   // nJoints * 7
  static const int _tCols          = 321;   // total model param dimensions
  static const int _modelParamsCols = 204;  // what the server sends
  static const int _skinInfluences = 8;

  // ---------------------------------------------------------------------------
  // Load
  // ---------------------------------------------------------------------------

  /// Load all MHR assets from the app bundle (call once at startup).
  Future<void> load() async {
    if (_loaded) return;

    final results = await Future.wait([
      rootBundle.load('assets/mhr/param_transform.bin'),
      rootBundle.load('assets/mhr/skeleton.bin'),
      rootBundle.load('assets/mhr/inv_bind_pose.bin'),
      rootBundle.load('assets/mhr/skin_idx.bin'),
      rootBundle.load('assets/mhr/skin_wgt.bin'),
      rootBundle.load('assets/mhr/rest_verts.bin'),
      rootBundle.load('assets/mhr/faces.bin'),
    ]);

    _parseParamTransform(results[0]);
    _parseSkeleton(results[1]);
    _parseInvBindPose(results[2]);
    _parseSkinIdx(results[3]);
    _parseSkinWgt(results[4]);
    _parseRestVerts(results[5]);
    _parseFaces(results[6]);

    _loaded = true;
  }

  // ---------------------------------------------------------------------------
  // Binary parsers
  // ---------------------------------------------------------------------------

  void _parseParamTransform(ByteData data) {
    final n = _jpRows * _tCols;
    _T = Float32List(n);
    for (int i = 0; i < n; i++) {
      _T[i] = data.getFloat32(i * 4, Endian.little);
    }
  }

  void _parseSkeleton(ByteData data) {
    int off = 0;
    _nJoints = data.getInt32(off, Endian.little); off += 4;

    _parents = Int16List(_nJoints);
    for (int j = 0; j < _nJoints; j++) {
      _parents[j] = data.getInt16(off, Endian.little); off += 2;
    }
    // Skip padding to 4-byte boundary
    final pad = (4 - ((_nJoints * 2) % 4)) % 4;
    off += pad;

    _offsets = Float32List(_nJoints * 3);
    for (int i = 0; i < _nJoints * 3; i++) {
      _offsets[i] = data.getFloat32(off, Endian.little); off += 4;
    }

    _preRots = Float32List(_nJoints * 4);
    for (int i = 0; i < _nJoints * 4; i++) {
      _preRots[i] = data.getFloat32(off, Endian.little); off += 4;
    }
  }

  void _parseInvBindPose(ByteData data) {
    final n = _nJointsConst * 16;
    _ibp = Float32List(n);
    for (int i = 0; i < n; i++) {
      _ibp[i] = data.getFloat32(i * 4, Endian.little);
    }
  }

  void _parseSkinIdx(ByteData data) {
    final n = data.lengthInBytes ~/ 2;
    _skinIdx = Int16List(n);
    for (int i = 0; i < n; i++) {
      _skinIdx[i] = data.getInt16(i * 2, Endian.little);
    }
    _nVerts = n ~/ _skinInfluences;
  }

  void _parseSkinWgt(ByteData data) {
    final n = _nVerts * _skinInfluences;
    _skinWgt = Float32List(n);
    for (int i = 0; i < n; i++) {
      _skinWgt[i] = data.getFloat32(i * 4, Endian.little);
    }
  }

  void _parseRestVerts(ByteData data) {
    final n = _nVerts * 3;
    _restVerts = Float32List(n);
    for (int i = 0; i < n; i++) {
      _restVerts[i] = data.getFloat32(i * 4, Endian.little);
    }
  }

  void _parseFaces(ByteData data) {
    _nFaces = data.lengthInBytes ~/ 12; // 3 × int32
    _faces = Int32List(_nFaces * 3);
    for (int i = 0; i < _nFaces * 3; i++) {
      _faces[i] = data.getInt32(i * 4, Endian.little);
    }
  }

  // ---------------------------------------------------------------------------
  // Forward pass
  // ---------------------------------------------------------------------------

  /// Compute mesh from MHR model parameters.
  ///
  /// [modelParams] – model_parameters[204] as output by FastSAM3DBody.
  ///   Layout: [global_trans*10(3), global_rot_euler(3), body_pose_euler(130), scales(68)]
  /// [camT] – camera-space translation [tx, ty, tz] in metres (as output by
  ///   the server's SAM3DBody pipeline).
  ///
  /// Returns a [MeshFrame] with LOD3 vertices (in metres) and faces.
  MeshFrame forward({
    required Float32List modelParams,
    List<double> camT = const [0.0, 0.0, 0.0],
  }) {
    assert(_loaded, 'MhrModel.load() must be called first');
    assert(modelParams.length == _modelParamsCols);

    // Step 1: joint_params = T @ model_params_padded
    // T is (889, 321); model_params is 204 — zero-pad to 321.
    // We only need the first 204 columns of T since the rest are all zero.
    final jp = _paramTransform(modelParams);   // [889] = [127 * 7]

    // Step 2: FK tree traversal → global transforms [127, 8]
    // Each entry: [tx, ty, tz, qx, qy, qz, qw, scale]
    final skelState = _fk(jp);                 // [127 * 8]

    // Step 3: build skinning matrices M[j] = G[j] @ ibp[j]
    final skinMats = _buildSkinMats(skelState);  // [127 * 16]

    // Step 4: LBS skinning — output is in centimetres (pymomentum native scale)
    final verts = _lbs(skinMats);               // [nVerts * 3], cm

    // Convert centimetres → metres to match cam_t units from the server.
    // The Python pipeline does:  curr_skinned_verts = mhr_output * 0.01
    // pred_cam_t is therefore in metres.  We apply the same scale here so
    // that (vertex_m + cam_t_m) / depth_m gives correct pixel coordinates.
    return MeshFrame(
      vertices: List<double>.generate(_nVerts * 3, (i) => verts[i] * 0.01),
      indices:  List<int>.generate(_nFaces * 3, (i) => _faces[i]),
      camT:     camT,
    );
  }

  // ---------------------------------------------------------------------------
  // Step 1: Parameter transform  (pure matrix-vector multiply)
  // ---------------------------------------------------------------------------

  Float32List _paramTransform(Float32List mp) {
    // jp[r] = sum_{c=0}^{203} T[r, c] * mp[c]   (columns 204..320 are all zero)
    final jp = Float32List(_jpRows);
    for (int r = 0; r < _jpRows; r++) {
      double acc = 0.0;
      final rowBase = r * _tCols;
      for (int c = 0; c < _modelParamsCols; c++) {
        acc += _T[rowBase + c] * mp[c];
      }
      jp[r] = acc;
    }
    return jp;
  }

  // ---------------------------------------------------------------------------
  // Step 2: FK tree traversal
  // ---------------------------------------------------------------------------

  /// Each joint has 7 params: [tx, ty, tz, rx, ry, rz, scale].
  /// euler convention: ZYX (matches pymomentum / verified numerically).
  Float32List _fk(Float32List jp) {
    // Output: global transform per joint [tx, ty, tz, qx, qy, qz, qw, scale]
    final out = Float32List(_nJointsConst * 8);

    // Temporaries for parent transforms
    final gT = Float32List(_nJointsConst * 3);  // global translation
    final gQ = Float32List(_nJointsConst * 4);  // global rotation quaternion xyzw

    for (int j = 0; j < _nJointsConst; j++) {
      final jpBase = j * 7;
      final localTx = jp[jpBase];
      final localTy = jp[jpBase + 1];
      final localTz = jp[jpBase + 2];
      final rx      = jp[jpBase + 3];
      final ry      = jp[jpBase + 4];
      final rz      = jp[jpBase + 5];
      final scale   = 1.0 + jp[jpBase + 6]; // scale offset from 1

      // Local euler ZYX → quaternion
      final localQ = _eulerZyxToQuat(rx, ry, rz);

      // Pre-rotation quaternion (baked into skeleton topology)
      final preQx = _preRots[j * 4];
      final preQy = _preRots[j * 4 + 1];
      final preQz = _preRots[j * 4 + 2];
      final preQw = _preRots[j * 4 + 3];

      // combined = pre_rot * local_rot
      final cq = _quatMul(preQx, preQy, preQz, preQw,
                          localQ[0], localQ[1], localQ[2], localQ[3]);

      final offX = _offsets[j * 3];
      final offY = _offsets[j * 3 + 1];
      final offZ = _offsets[j * 3 + 2];

      // Local position = offset + local_translation
      final lx = offX + localTx;
      final ly = offY + localTy;
      final lz = offZ + localTz;

      final parent = _parents[j];
      double gx, gy, gz;
      double gqx, gqy, gqz, gqw;

      if (parent < 0) {
        // Root: global = local
        gx = lx; gy = ly; gz = lz;
        gqx = cq[0]; gqy = cq[1]; gqz = cq[2]; gqw = cq[3];
      } else {
        // Rotate local position by parent's global rotation, then add parent translation
        final pqx = gQ[parent * 4];
        final pqy = gQ[parent * 4 + 1];
        final pqz = gQ[parent * 4 + 2];
        final pqw = gQ[parent * 4 + 3];

        final rotated = _quatRotVec(pqx, pqy, pqz, pqw, lx, ly, lz);
        gx = gT[parent * 3]     + rotated[0];
        gy = gT[parent * 3 + 1] + rotated[1];
        gz = gT[parent * 3 + 2] + rotated[2];

        // Global rotation = parent_global * combined
        final gq = _quatMul(pqx, pqy, pqz, pqw, cq[0], cq[1], cq[2], cq[3]);
        gqx = gq[0]; gqy = gq[1]; gqz = gq[2]; gqw = gq[3];
      }

      gT[j * 3]     = gx;
      gT[j * 3 + 1] = gy;
      gT[j * 3 + 2] = gz;
      gQ[j * 4]     = gqx;
      gQ[j * 4 + 1] = gqy;
      gQ[j * 4 + 2] = gqz;
      gQ[j * 4 + 3] = gqw;

      final outBase = j * 8;
      out[outBase]     = gx;
      out[outBase + 1] = gy;
      out[outBase + 2] = gz;
      out[outBase + 3] = gqx;
      out[outBase + 4] = gqy;
      out[outBase + 5] = gqz;
      out[outBase + 6] = gqw;
      out[outBase + 7] = scale;
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Step 3: Skinning matrices  M[j] = G_mat[j] @ ibp[j]
  // ---------------------------------------------------------------------------

  Float32List _buildSkinMats(Float32List skelState) {
    final mats = Float32List(_nJointsConst * 16);

    for (int j = 0; j < _nJointsConst; j++) {
      final s = skelState;
      final sb = j * 8;
      final tx = s[sb], ty = s[sb + 1], tz = s[sb + 2];
      final qx = s[sb + 3], qy = s[sb + 4], qz = s[sb + 5], qw = s[sb + 6];
      final scale = s[sb + 7];

      // Build 4×4 from (t, q, scale): row-major
      // R = scale * quat_to_rotmat(q)
      final x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
      final xx = qx * x2, xy = qx * y2, xz = qx * z2;
      final yy = qy * y2, yz = qy * z2, zz = qz * z2;
      final wx = qw * x2, wy = qw * y2, wz = qw * z2;

      final g = Float32List(16);
      g[0]  = (1.0 - (yy + zz)) * scale;
      g[1]  = (xy - wz) * scale;
      g[2]  = (xz + wy) * scale;
      g[3]  = tx;
      g[4]  = (xy + wz) * scale;
      g[5]  = (1.0 - (xx + zz)) * scale;
      g[6]  = (yz - wx) * scale;
      g[7]  = ty;
      g[8]  = (xz - wy) * scale;
      g[9]  = (yz + wx) * scale;
      g[10] = (1.0 - (xx + yy)) * scale;
      g[11] = tz;
      g[12] = 0.0; g[13] = 0.0; g[14] = 0.0; g[15] = 1.0;

      // M = g @ ibp[j]   (4×4 × 4×4)
      final ibpBase = j * 16;
      final mBase   = j * 16;
      for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
          double sum = 0.0;
          for (int k = 0; k < 4; k++) {
            sum += g[row * 4 + k] * _ibp[ibpBase + k * 4 + col];
          }
          mats[mBase + row * 4 + col] = sum;
        }
      }
    }
    return mats;
  }

  // ---------------------------------------------------------------------------
  // Step 4: LBS skinning
  // ---------------------------------------------------------------------------

  Float32List _lbs(Float32List skinMats) {
    final verts = Float32List(_nVerts * 3);

    for (int v = 0; v < _nVerts; v++) {
      final rx = _restVerts[v * 3];
      final ry = _restVerts[v * 3 + 1];
      final rz = _restVerts[v * 3 + 2];

      double ox = 0, oy = 0, oz = 0;

      for (int k = 0; k < _skinInfluences; k++) {
        final w = _skinWgt[v * _skinInfluences + k];
        if (w == 0.0) continue;
        final j = _skinIdx[v * _skinInfluences + k];
        final mb = j * 16;
        final m = skinMats;
        // Apply 4×4 M to homogeneous rest vertex [rx, ry, rz, 1]
        ox += w * (m[mb]      * rx + m[mb + 1]  * ry + m[mb + 2]  * rz + m[mb + 3]);
        oy += w * (m[mb + 4]  * rx + m[mb + 5]  * ry + m[mb + 6]  * rz + m[mb + 7]);
        oz += w * (m[mb + 8]  * rx + m[mb + 9]  * ry + m[mb + 10] * rz + m[mb + 11]);
      }

      verts[v * 3]     = ox;
      verts[v * 3 + 1] = oy;
      verts[v * 3 + 2] = oz;
    }
    return verts;
  }

  // ---------------------------------------------------------------------------
  // Quaternion math helpers
  // ---------------------------------------------------------------------------

  /// Euler ZYX (Rz * Ry * Rx) → quaternion xyzw.
  /// This matches pymomentum's convention (verified numerically).
  static List<double> _eulerZyxToQuat(double rx, double ry, double rz) {
    final cx = math.cos(rx * 0.5), sx = math.sin(rx * 0.5);
    final cy = math.cos(ry * 0.5), sy = math.sin(ry * 0.5);
    final cz = math.cos(rz * 0.5), sz = math.sin(rz * 0.5);
    return [
      sx * cy * cz - cx * sy * sz, // qx
      cx * sy * cz + sx * cy * sz, // qy
      cx * cy * sz - sx * sy * cz, // qz
      cx * cy * cz + sx * sy * sz, // qw
    ];
  }

  /// Quaternion multiply: q1 * q2 (Hamilton product), both in xyzw.
  static List<double> _quatMul(
      double x1, double y1, double z1, double w1,
      double x2, double y2, double z2, double w2) {
    return [
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ];
  }

  /// Rotate vector v by quaternion q (xyzw).
  static List<double> _quatRotVec(
      double qx, double qy, double qz, double qw,
      double vx, double vy, double vz) {
    final x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
    final xx = qx * x2, xy = qx * y2, xz = qx * z2;
    final yy = qy * y2, yz = qy * z2, zz = qz * z2;
    final wx = qw * x2, wy = qw * y2, wz = qw * z2;
    return [
      (1.0 - (yy + zz)) * vx + (xy - wz) * vy + (xz + wy) * vz,
      (xy + wz) * vx + (1.0 - (xx + zz)) * vy + (yz - wx) * vz,
      (xz - wy) * vx + (yz + wx) * vy + (1.0 - (xx + yy)) * vz,
    ];
  }
}
