import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:vector_math/vector_math_64.dart' as vm;

import '../models/mesh_frame.dart';

/// A widget that renders a 3D triangle mesh using Flutter's Canvas (CustomPaint).
///
/// Features:
/// - Orbit rotation via touch-drag
/// - Pinch-to-zoom (via GestureDetector)
/// - Simple directional lighting
///
/// This approach uses software rasterization through Canvas, which works on all
/// platforms without requiring OpenGL. For high-poly meshes on the real
/// SAM3DBody output (~18k verts, ~37k tris) this will be adequately performant
/// for scrubbing; for real-time playback a GPU-backed renderer (flutter_gl) is
/// recommended.
class MeshViewer extends StatefulWidget {
  final MeshFrame frame;

  /// When true: gestures are disabled and the mesh is projected using the
  /// camera translation stored in [frame.camT] + [focalLength].
  final bool overlay;

  /// Focal length in pixels (from the server header).  Used only in overlay
  /// mode.  0 means "estimate from widget size".
  final double focalLength;

  const MeshViewer({
    super.key,
    required this.frame,
    this.overlay = false,
    this.focalLength = 0.0,
  });

  @override
  State<MeshViewer> createState() => _MeshViewerState();
}

class _MeshViewerState extends State<MeshViewer> {
  // Default orientation: no rotation.  mhr2smpl outputs go in Y-up /
  // Z-toward-viewer space (after the [1,2]*=-1 flip in mesh_gen.py), so the
  // rest pose already appears upright and front-facing with rotX = 0.
  static const double _defaultRotX = 0.0;
  static const double _defaultRotY = 0.0;
  static const double _defaultZoom = 1.0;

  double _rotX = _defaultRotX;
  double _rotY = _defaultRotY;
  double _zoom = _defaultZoom;

  Offset? _lastFocalPoint;
  double _baseZoom = 1.0;

  void _resetOrientation() {
    setState(() {
      _rotX = _defaultRotX;
      _rotY = _defaultRotY;
      _zoom = _defaultZoom;
    });
  }

  @override
  Widget build(BuildContext context) {
    // In overlay mode: no interaction, no reset button, use perspective painter.
    if (widget.overlay) {
      return CustomPaint(
        painter: _MeshPainterProjected(
          frame: widget.frame,
          focalLength: widget.focalLength,
        ),
        size: Size.infinite,
      );
    }

    return Stack(
      children: [
        GestureDetector(
          onScaleStart: (d) {
            _lastFocalPoint = d.localFocalPoint;
            _baseZoom = _zoom;
          },
          onScaleUpdate: (d) {
            setState(() {
              // Orbit from focal point delta
              if (_lastFocalPoint != null) {
                final dx = d.localFocalPoint.dx - _lastFocalPoint!.dx;
                final dy = d.localFocalPoint.dy - _lastFocalPoint!.dy;
                _rotY += dx * 0.01;
                _rotX += dy * 0.01;
                // No clamp — allow full 360° rotation in all axes
              }
              _lastFocalPoint = d.localFocalPoint;
              // Pinch zoom
              _zoom = (_baseZoom * d.scale).clamp(0.2, 5.0);
            });
          },
          onScaleEnd: (_) => _lastFocalPoint = null,
          child: CustomPaint(
            painter: _MeshPainter(
              frame: widget.frame,
              rotX: _rotX,
              rotY: _rotY,
              zoom: _zoom,
            ),
            size: Size.infinite,
          ),
        ),
        // Reset orientation button
        Positioned(
          top: 8,
          right: 8,
          child: IconButton(
            onPressed: _resetOrientation,
            icon: const Icon(Icons.crop_rotate, color: Colors.white70),
            tooltip: 'Reset orientation',
            style: IconButton.styleFrom(
              backgroundColor: Colors.black38,
            ),
          ),
        ),
      ],
    );
  }
}

class _MeshPainter extends CustomPainter {
  final MeshFrame frame;
  final double rotX;
  final double rotY;
  final double zoom;

  _MeshPainter({
    required this.frame,
    required this.rotX,
    required this.rotY,
    required this.zoom,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (frame.vertices.isEmpty || frame.indices.isEmpty) {
      // Draw placeholder text
      final tp = TextPainter(
        text: const TextSpan(
          text: 'No mesh data',
          style: TextStyle(color: Colors.white54, fontSize: 18),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(size.width / 2 - tp.width / 2, size.height / 2 - tp.height / 2));
      return;
    }

    final cx = size.width / 2;
    final cy = size.height / 2;
    final scale = math.min(cx, cy) * 0.8 * zoom;

    // Build rotation matrix (Y then X)
    final rotMatY = vm.Matrix4.rotationY(rotY);
    final rotMatX = vm.Matrix4.rotationX(rotX);
    final rotMat = rotMatX.multiplied(rotMatY);

    // Compute mesh center for centering
    final meshCenter = frame.center;
    final mc = vm.Vector3(meshCenter[0], meshCenter[1], meshCenter[2]);

    // Project vertices
    final n = frame.vertexCount;
    final projX = Float32List(n);
    final projY = Float32List(n);
    final projZ = Float32List(n);

    // Bounding radius for normalization
    final br = frame.boundingRadius;
    final normScale = br > 1e-8 ? 1.0 / br : 1.0;

    for (int i = 0; i < n; i++) {
      final vx = (frame.vertices[i * 3] - mc.x) * normScale;
      final vy = (frame.vertices[i * 3 + 1] - mc.y) * normScale;
      final vz = (frame.vertices[i * 3 + 2] - mc.z) * normScale;

      final v = vm.Vector3(vx, vy, vz);
      rotMat.transform3(v);

      projX[i] = cx + v.x * scale;
      projY[i] = cy - v.y * scale; // flip Y: FK verts are Y-up, screen is Y-down
      projZ[i] = v.z;
    }

    // Light direction (fixed in view space)
    final lightDir = vm.Vector3(0.3, 0.6, 1.0)..normalize();

    // Collect triangles with depth for painter's algorithm sorting
    final triCount = frame.indices.length ~/ 3;
    final triDepth = Float32List(triCount);
    final triOrder = List<int>.generate(triCount, (i) => i);

    for (int t = 0; t < triCount; t++) {
      final i0 = frame.indices[t * 3];
      final i1 = frame.indices[t * 3 + 1];
      final i2 = frame.indices[t * 3 + 2];
      triDepth[t] = (projZ[i0] + projZ[i1] + projZ[i2]) / 3.0;
    }

    // Sort back-to-front
    triOrder.sort((a, b) => triDepth[a].compareTo(triDepth[b]));

    // Draw triangles
    final path = ui.Path();
    final paint = Paint()..style = PaintingStyle.fill;

    const baseColor = Color(0xFF5C8FE6);

    for (final t in triOrder) {
      final i0 = frame.indices[t * 3];
      final i1 = frame.indices[t * 3 + 1];
      final i2 = frame.indices[t * 3 + 2];

      // Compute face normal from 3D positions for lighting
      final vx0 = frame.vertices[i0 * 3], vy0 = frame.vertices[i0 * 3 + 1], vz0 = frame.vertices[i0 * 3 + 2];
      final vx1 = frame.vertices[i1 * 3], vy1 = frame.vertices[i1 * 3 + 1], vz1 = frame.vertices[i1 * 3 + 2];
      final vx2 = frame.vertices[i2 * 3], vy2 = frame.vertices[i2 * 3 + 1], vz2 = frame.vertices[i2 * 3 + 2];

      final e1 = vm.Vector3(vx1 - vx0, vy1 - vy0, vz1 - vz0);
      final e2 = vm.Vector3(vx2 - vx0, vy2 - vy0, vz2 - vz0);
      final faceNormal = e1.cross(e2)..normalize();

      // Transform normal by rotation
      rotMat.transform3(faceNormal);

      // Simple diffuse lighting + ambient
      final diffuse = math.max(0.0, faceNormal.dot(lightDir));
      const ambient = 0.3;
      final brightness = (ambient + diffuse * 0.7).clamp(0.0, 1.0);

      final r = ((baseColor.r * 255.0).round() * brightness).round().clamp(0, 255);
      final g = ((baseColor.g * 255.0).round() * brightness).round().clamp(0, 255);
      final b = ((baseColor.b * 255.0).round() * brightness).round().clamp(0, 255);

      paint.color = Color.fromARGB(255, r, g, b);

      path.reset();
      path.moveTo(projX[i0], projY[i0]);
      path.lineTo(projX[i1], projY[i1]);
      path.lineTo(projX[i2], projY[i2]);
      path.close();

      canvas.drawPath(path, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _MeshPainter old) {
    return old.frame != frame || old.rotX != rotX || old.rotY != rotY || old.zoom != zoom;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Perspective projection painter  (overlay mode)
// ─────────────────────────────────────────────────────────────────────────────

/// Projects SMPL vertices onto the canvas using the camera translation stored
/// in [frame.camT] and the focal length from the server.
///
/// Coordinate convention (matches the SMPL FK output):
///   X – right,  Y – up,  Z – toward viewer
///
/// Perspective projection:
///   x_screen = fx * (X + tx) / (Z + tz) + cx
///   y_screen = cy - fy * (Y + ty) / (Z + tz)   ← flip Y for screen
class _MeshPainterProjected extends CustomPainter {
  final MeshFrame frame;
  final double focalLength;

  _MeshPainterProjected({required this.frame, required this.focalLength});

  @override
  void paint(Canvas canvas, Size size) {
    if (frame.vertices.isEmpty || frame.indices.isEmpty) return;

    final cx = size.width  / 2;
    final cy = size.height / 2;

    // Use provided focal length or fall back to an estimate based on a 60° FOV.
    final fx = focalLength > 0
        ? focalLength
        : size.width / (2 * math.tan(math.pi / 6)); // 60° horiz FOV
    final fy = fx; // square pixels assumed

    final tx = frame.camT[0];
    final ty = frame.camT[1];
    final tz = frame.camT[2];

    final n = frame.vertexCount;
    final projX = Float32List(n);
    final projY = Float32List(n);
    final projZ = Float32List(n);

    for (int i = 0; i < n; i++) {
      final X = frame.vertices[i * 3];
      final Y = frame.vertices[i * 3 + 1];
      final Z = frame.vertices[i * 3 + 2];

      final dz = Z + tz;
      if (dz.abs() < 1e-6) {
        projX[i] = cx;
        projY[i] = cy;
        projZ[i] = 0;
        continue;
      }
      projX[i] = (fx * (X + tx) / dz + cx).toDouble();
      projY[i] = (cy - fy * (Y + ty) / dz).toDouble();
      projZ[i] = dz;
    }

    final lightDir = vm.Vector3(0.3, 0.6, 1.0)..normalize();
    final triCount = frame.indices.length ~/ 3;
    final triDepth = Float32List(triCount);
    final triOrder = List<int>.generate(triCount, (i) => i);

    for (int t = 0; t < triCount; t++) {
      final i0 = frame.indices[t * 3];
      final i1 = frame.indices[t * 3 + 1];
      final i2 = frame.indices[t * 3 + 2];
      triDepth[t] = (projZ[i0] + projZ[i1] + projZ[i2]) / 3.0;
    }
    triOrder.sort((a, b) => triDepth[b].compareTo(triDepth[a])); // far→near

    final path  = ui.Path();
    final paint = Paint()..style = PaintingStyle.fill;
    const baseColor = Color(0xFF5C8FE6);

    for (final t in triOrder) {
      final i0 = frame.indices[t * 3];
      final i1 = frame.indices[t * 3 + 1];
      final i2 = frame.indices[t * 3 + 2];

      final vx0 = frame.vertices[i0 * 3], vy0 = frame.vertices[i0 * 3 + 1], vz0 = frame.vertices[i0 * 3 + 2];
      final vx1 = frame.vertices[i1 * 3], vy1 = frame.vertices[i1 * 3 + 1], vz1 = frame.vertices[i1 * 3 + 2];
      final vx2 = frame.vertices[i2 * 3], vy2 = frame.vertices[i2 * 3 + 1], vz2 = frame.vertices[i2 * 3 + 2];

      final e1 = vm.Vector3(vx1 - vx0, vy1 - vy0, vz1 - vz0);
      final e2 = vm.Vector3(vx2 - vx0, vy2 - vy0, vz2 - vz0);
      final faceNormal = e1.cross(e2)..normalize();

      final diffuse = math.max(0.0, faceNormal.dot(lightDir));
      const ambient = 0.3;
      final brightness = (ambient + diffuse * 0.7).clamp(0.0, 1.0);

      final r = ((baseColor.r * 255.0).round() * brightness).round().clamp(0, 255);
      final g = ((baseColor.g * 255.0).round() * brightness).round().clamp(0, 255);
      final b = ((baseColor.b * 255.0).round() * brightness).round().clamp(0, 255);
      paint.color = Color.fromARGB(255, r, g, b);

      path.reset();
      path.moveTo(projX[i0], projY[i0]);
      path.lineTo(projX[i1], projY[i1]);
      path.lineTo(projX[i2], projY[i2]);
      path.close();
      canvas.drawPath(path, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _MeshPainterProjected old) =>
      old.frame != frame || old.focalLength != focalLength;
}
