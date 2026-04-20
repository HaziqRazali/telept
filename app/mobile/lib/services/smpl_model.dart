/// SMPL body model forward kinematics implemented in Dart.
///
/// Loads smpl_neutral.npz from Flutter assets and computes per-vertex
/// positions given global_orient[3], body_pose[63], betas[10].
///
/// Mathematical pipeline:
///   1. Shape blend shapes:  v_shaped = v_template + shapedirs @ betas
///   2. Joint locations:     J = J_regressor @ v_shaped
///   3. Pose: axis-angle → rotation matrices (Rodrigues)
///   4. FK:  world joint transforms via kinematic tree
///   5. LBS: v_posed = sum_j( w_j * (T_j @ v_shaped_h) )
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:flutter/services.dart';

import '../models/mesh_frame.dart';

class SmplModel {
  // Arrays extracted from smpl_neutral.npz
  late final Float32List _vTemplate;    // [6890*3]
  late final Float32List _shapedirs;    // [6890*3*10]
  late final Float32List _jRegressor;   // [24*6890]
  late final Float32List _lbsWeights;   // [6890*24]
  late final Int32List   _kintree;      // [24]  parent indices (-1 for root)
  late final Int32List   _faces;        // [13776*3]

  static const int _nVerts   = 6890;
  static const int _nJoints  = 24;
  static const int _nBetas   = 10;
  static const int _nFaces   = 13776;

  bool _loaded = false;

  static SmplModel? _instance;
  static SmplModel get instance => _instance ??= SmplModel._();
  SmplModel._();

  /// Load the model from assets (call once at startup).
  Future<void> load() async {
    if (_loaded) return;
    final ByteData data = await rootBundle.load('assets/smpl_neutral.npz');
    final bytes = data.buffer.asUint8List();
    _parseNpz(bytes);
    _loaded = true;
  }

  // ---------------------------------------------------------------------------
  // NPZ parser (ZIP of .npy files)
  // ---------------------------------------------------------------------------
  void _parseNpz(Uint8List bytes) {
    final archive = ZipDecoder().decodeBytes(bytes);
    final arrays = <String, dynamic>{};
    for (final f in archive) {
      if (!f.isFile || !f.name.endsWith('.npy')) continue;
      final name = f.name.replaceAll('.npy', '');
      final content = f.content as List<int>;
      arrays[name] = _parseNpy(Uint8List.fromList(content));
    }
    _vTemplate   = arrays['v_template']   as Float32List;
    _shapedirs   = arrays['shapedirs']    as Float32List;
    _jRegressor  = arrays['J_regressor']  as Float32List;
    _lbsWeights  = arrays['lbs_weights']  as Float32List;
    _faces       = arrays['faces']        as Int32List;

    // kintree_table: parent indices as int32
    final rawKintree = arrays['kintree_table'];
    if (rawKintree is Int32List) {
      _kintree = rawKintree;
    } else if (rawKintree is Int64List) {
      _kintree = Int32List.fromList(rawKintree.map((v) => v.toInt()).toList());
    } else {
      _kintree = Int32List.fromList((rawKintree as List).map((v) => (v as num).toInt()).toList());
    }
  }

  /// Minimal .npy parser supporting float32, float64, int32, int64, C/F order.
  static dynamic _parseNpy(Uint8List bytes) {
    // Magic: \x93NUMPY
    if (bytes[0] != 0x93 || bytes[1] != 0x4E) {
      throw FormatException('Not a valid .npy file');
    }
    final majorVersion = bytes[6];
    final headerLen = majorVersion >= 2
        ? ByteData.sublistView(bytes, 8, 12).getUint32(0, Endian.little)
        : ByteData.sublistView(bytes, 8, 10).getUint16(0, Endian.little);
    final headerStart = majorVersion >= 2 ? 12 : 10;
    final headerBytes = bytes.sublist(headerStart, headerStart + headerLen);
    final header = String.fromCharCodes(headerBytes);

    // Parse dtype
    final dtypeMatch = RegExp(r"'descr'\s*:\s*'([^']+)'").firstMatch(header);
    final dtype = dtypeMatch!.group(1)!;

    // Parse shape
    final shapeMatch = RegExp(r"'shape'\s*:\s*\(([^)]*)\)").firstMatch(header);
    final shapeStr = shapeMatch!.group(1)!.trim();
    final shape = shapeStr.isEmpty
        ? <int>[]
        : shapeStr.split(',').map((s) => s.trim()).where((s) => s.isNotEmpty).map(int.parse).toList();

    // Parse fortran order
    final fortranMatch = RegExp(r"'fortran_order'\s*:\s*(True|False)").firstMatch(header);
    final fortranOrder = fortranMatch?.group(1) == 'True';

    final dataStart = headerStart + headerLen;
    final dataBytes = bytes.sublist(dataStart);
    final totalElements = shape.isEmpty ? 1 : shape.fold(1, (a, b) => a * b);

    dynamic flat;
    if (dtype.contains('f4') || dtype.contains('f8')) {
      flat = _readFloats(dataBytes, totalElements, dtype.contains('f8'));
    } else if (dtype.contains('i4') || dtype.contains('i8') || dtype.contains('u4') || dtype.contains('u8')) {
      flat = _readInts(dataBytes, totalElements, dtype);
    } else {
      throw FormatException('Unsupported dtype: $dtype');
    }

    // Handle Fortran order by transposing if needed
    if (fortranOrder && shape.length > 1) {
      flat = _fortranToC(flat, shape);
    }

    return flat;
  }

  static Float32List _readFloats(Uint8List bytes, int n, bool isFloat64) {
    if (isFloat64) {
      final bd = ByteData.sublistView(bytes);
      final out = Float32List(n);
      for (int i = 0; i < n; i++) {
        out[i] = bd.getFloat64(i * 8, Endian.little).toDouble();
      }
      return out;
    }
    // float32 — can view directly if aligned
    final bd = ByteData.sublistView(bytes);
    final out = Float32List(n);
    for (int i = 0; i < n; i++) {
      out[i] = bd.getFloat32(i * 4, Endian.little);
    }
    return out;
  }

  static dynamic _readInts(Uint8List bytes, int n, String dtype) {
    final bd = ByteData.sublistView(bytes);
    if (dtype.contains('i4') || dtype.contains('u4')) {
      final out = Int32List(n);
      for (int i = 0; i < n; i++) {
        out[i] = bd.getInt32(i * 4, Endian.little);
      }
      return out;
    } else {
      final out = Int64List(n);
      for (int i = 0; i < n; i++) {
        out[i] = bd.getInt64(i * 8, Endian.little);
      }
      return out;
    }
  }

  static Float32List _fortranToC(Float32List flat, List<int> shape) {
    // Convert Fortran (column-major) layout to C (row-major)
    final n = flat.length;
    final out = Float32List(n);
    // Compute strides for Fortran order
    final ndim = shape.length;
    final fStrides = List<int>.filled(ndim, 1);
    for (int i = 1; i < ndim; i++) fStrides[i] = fStrides[i - 1] * shape[i - 1];
    // Compute strides for C order
    final cStrides = List<int>.filled(ndim, 1);
    for (int i = ndim - 2; i >= 0; i--) cStrides[i] = cStrides[i + 1] * shape[i + 1];

    final indices = List<int>.filled(ndim, 0);
    for (int ci = 0; ci < n; ci++) {
      // ci → indices (C order)
      int tmp = ci;
      for (int d = 0; d < ndim; d++) {
        indices[d] = tmp ~/ cStrides[d];
        tmp %= cStrides[d];
      }
      // indices → Fortran flat index
      int fi = 0;
      for (int d = 0; d < ndim; d++) fi += indices[d] * fStrides[d];
      out[ci] = flat[fi];
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // SMPL Forward Kinematics
  // ---------------------------------------------------------------------------

  /// Compute mesh from SMPL parameters.
  ///
  /// [go]       – global orient, axis-angle [3]
  /// [bodyPose] – body pose, axis-angle [63] (21 joints × 3)
  /// [betas]    – shape coefficients [10]
  ///
  /// Returns a [MeshFrame] with 6890 vertices and 13776 faces.
  MeshFrame forward({
    required List<double> go,
    required List<double> bodyPose,
    required List<double> betas,
  }) {
    assert(_loaded, 'SmplModel.load() must be called first');

    // 1. Shape blend shapes: v_shaped = v_template + shapedirs @ betas
    //    shapedirs: [6890, 3, 10]  →  each vertex, each axis, each beta
    final vShaped = Float32List(_nVerts * 3);
    for (int v = 0; v < _nVerts; v++) {
      for (int ax = 0; ax < 3; ax++) {
        double val = _vTemplate[v * 3 + ax];
        for (int b = 0; b < _nBetas; b++) {
          val += _shapedirs[(v * 3 + ax) * _nBetas + b] * betas[b];
        }
        vShaped[v * 3 + ax] = val.toDouble();
      }
    }

    // 2. Joint locations: J = J_regressor @ v_shaped
    //    J_regressor: [24, 6890]
    final joints = Float32List(_nJoints * 3);
    for (int j = 0; j < _nJoints; j++) {
      double jx = 0, jy = 0, jz = 0;
      final rowOffset = j * _nVerts;
      for (int v = 0; v < _nVerts; v++) {
        final w = _jRegressor[rowOffset + v];
        if (w == 0.0) continue;
        jx += w * vShaped[v * 3];
        jy += w * vShaped[v * 3 + 1];
        jz += w * vShaped[v * 3 + 2];
      }
      joints[j * 3]     = jx.toDouble();
      joints[j * 3 + 1] = jy.toDouble();
      joints[j * 3 + 2] = jz.toDouble();
    }

    // 3. Build full pose array: [go(3), bodyPose(63)] = 24 joints × 3
    final allPose = Float32List(_nJoints * 3);
    allPose[0] = go[0].toDouble();
    allPose[1] = go[1].toDouble();
    allPose[2] = go[2].toDouble();
    for (int i = 0; i < 63; i++) {
      allPose[3 + i] = bodyPose[i].toDouble();
    }

    // 4. Axis-angle → 3×3 rotation matrices (Rodrigues formula)
    //    Each joint: allPose[j*3 .. j*3+2] → R[j] (9 floats, row-major)
    final rotMats = Float32List(_nJoints * 9);
    for (int j = 0; j < _nJoints; j++) {
      final ax = allPose[j * 3];
      final ay = allPose[j * 3 + 1];
      final az = allPose[j * 3 + 2];
      _axisAngleToRotMat(ax, ay, az, rotMats, j * 9);
    }

    // 5. FK: compute world transform T[j] = [R|t] for each joint (4×4 matrices)
    //    T[j] = T[parent(j)] * T_local[j]
    //    where T_local[j] = [ R[j]  (J[j] - J[parent(j)]) ]
    //                       [ 0     1                       ]
    //    Then subtract the un-posed joint position so T maps rest→posed.
    //
    // We store each transform as a flat 16-float (row-major 4×4).
    final transforms = Float32List(_nJoints * 16);

    for (int j = 0; j < _nJoints; j++) {
      final parent = _kintree[j]; // -1 for root

      // Local rotation from pose
      final R = rotMats.sublist(j * 9, j * 9 + 9);

      // Joint offset from parent (in rest pose)
      double tx, ty, tz;
      if (parent < 0) {
        // Root: absolute position
        tx = joints[j * 3];
        ty = joints[j * 3 + 1];
        tz = joints[j * 3 + 2];
      } else {
        // Relative to parent
        tx = joints[j * 3]     - joints[parent * 3];
        ty = joints[j * 3 + 1] - joints[parent * 3 + 1];
        tz = joints[j * 3 + 2] - joints[parent * 3 + 2];
      }

      // Local T_local = [R | t]  (4×4)
      final local = Float32List(16);
      local[0]  = R[0]; local[1]  = R[1]; local[2]  = R[2];  local[3]  = tx;
      local[4]  = R[3]; local[5]  = R[4]; local[6]  = R[5];  local[7]  = ty;
      local[8]  = R[6]; local[9]  = R[7]; local[10] = R[8];  local[11] = tz;
      local[12] = 0;    local[13] = 0;    local[14] = 0;      local[15] = 1;

      if (parent < 0) {
        // Root: transform is local
        for (int k = 0; k < 16; k++) transforms[j * 16 + k] = local[k];
      } else {
        // Chain: T[j] = T[parent] * local
        _matMul4x4(transforms, parent * 16, local, 0, transforms, j * 16);
      }
    }

    // Subtract rest-pose joint contribution:
    // G[j] = T[j] * [I | -J[j]; 0 | 1]
    // so that vertices at their rest positions map correctly.
    final G = Float32List(_nJoints * 16);
    for (int j = 0; j < _nJoints; j++) {
      // compute T[j] @ [I | -J_rest[j]]
      final jx = joints[j * 3], jy = joints[j * 3 + 1], jz = joints[j * 3 + 2];
      // [I | -Jrest] as 4×4
      final sub = Float32List(16);
      sub[0] = 1; sub[5] = 1; sub[10] = 1; sub[15] = 1;
      sub[3] = -jx; sub[7] = -jy; sub[11] = -jz;
      _matMul4x4(transforms, j * 16, sub, 0, G, j * 16);
    }

    // 6. LBS: blend G[j] weighted by lbs_weights[v, j]
    //    v_posed[v] = sum_j( w[v,j] * G[j] @ [vShaped[v]; 1] )
    final vPosed = Float32List(_nVerts * 3);
    for (int v = 0; v < _nVerts; v++) {
      final vx = vShaped[v * 3];
      final vy = vShaped[v * 3 + 1];
      final vz = vShaped[v * 3 + 2];

      double ox = 0, oy = 0, oz = 0;
      for (int j = 0; j < _nJoints; j++) {
        final w = _lbsWeights[v * _nJoints + j];
        if (w < 1e-8) continue;
        final g = G;
        final off = j * 16;
        ox += w * (g[off]     * vx + g[off + 1] * vy + g[off + 2]  * vz + g[off + 3]);
        oy += w * (g[off + 4] * vx + g[off + 5] * vy + g[off + 6]  * vz + g[off + 7]);
        oz += w * (g[off + 8] * vx + g[off + 9] * vy + g[off + 10] * vz + g[off + 11]);
      }
      vPosed[v * 3]     = ox.toDouble();
      vPosed[v * 3 + 1] = oy.toDouble();
      vPosed[v * 3 + 2] = oz.toDouble();
    }

    return MeshFrame(
      vertices: List<double>.unmodifiable(vPosed),
      indices: List<int>.unmodifiable(_faces),
    );
  }

  // ---------------------------------------------------------------------------
  // Math helpers
  // ---------------------------------------------------------------------------

  /// Rodrigues formula: axis-angle (ax,ay,az) → 3×3 rotation matrix.
  /// Result written into [out] starting at [offset] (row-major, 9 floats).
  static void _axisAngleToRotMat(
      double ax, double ay, double az, Float32List out, int offset) {
    final theta2 = ax * ax + ay * ay + az * az;
    if (theta2 < 1e-10) {
      // Identity
      out[offset]     = 1; out[offset + 1] = 0; out[offset + 2] = 0;
      out[offset + 3] = 0; out[offset + 4] = 1; out[offset + 5] = 0;
      out[offset + 6] = 0; out[offset + 7] = 0; out[offset + 8] = 1;
      return;
    }
    final theta = math.sqrt(theta2);
    final ux = ax / theta, uy = ay / theta, uz = az / theta;
    final c = math.cos(theta);
    final s = math.sin(theta);
    final t = 1.0 - c;

    out[offset]     = t * ux * ux + c;
    out[offset + 1] = t * ux * uy - s * uz;
    out[offset + 2] = t * ux * uz + s * uy;
    out[offset + 3] = t * ux * uy + s * uz;
    out[offset + 4] = t * uy * uy + c;
    out[offset + 5] = t * uy * uz - s * ux;
    out[offset + 6] = t * ux * uz - s * uy;
    out[offset + 7] = t * uy * uz + s * ux;
    out[offset + 8] = t * uz * uz + c;
  }

  /// 4×4 matrix multiply: C = A * B (all row-major in flat arrays).
  static void _matMul4x4(
      Float32List a, int aOff, Float32List b, int bOff, Float32List c, int cOff) {
    for (int row = 0; row < 4; row++) {
      for (int col = 0; col < 4; col++) {
        double sum = 0;
        for (int k = 0; k < 4; k++) {
          sum += a[aOff + row * 4 + k] * b[bOff + k * 4 + col];
        }
        c[cOff + row * 4 + col] = sum.toDouble();
      }
    }
  }

  // Expose faces for reuse
  Int32List get faces => _faces;
  bool get isLoaded => _loaded;
}
