/// Parsed mesh data from a single OBJ frame.
class MeshFrame {
  /// Vertex positions – flat list [x0,y0,z0, x1,y1,z1, ...].
  final List<double> vertices;

  /// Triangle indices – flat list [i0,i1,i2, ...] (0-indexed).
  final List<int> indices;

  /// Vertex normals (per-vertex, flat). If empty, compute flat normals.
  final List<double> normals;

  /// Camera-space translation [tx, ty, tz] for perspective projection.
  /// [0, 0, 0] when not available (e.g. stub / invalid frame).
  final List<double> camT;

  const MeshFrame({
    required this.vertices,
    required this.indices,
    this.normals = const [],
    this.camT = const [0.0, 0.0, 0.0],
  });

  int get vertexCount => vertices.length ~/ 3;
  int get triangleCount => indices.length ~/ 3;

  /// Compute axis-aligned bounding-box center.
  List<double> get center {
    if (vertices.isEmpty) return [0, 0, 0];
    double mx = 0, my = 0, mz = 0;
    final n = vertexCount;
    for (int i = 0; i < n; i++) {
      mx += vertices[i * 3];
      my += vertices[i * 3 + 1];
      mz += vertices[i * 3 + 2];
    }
    return [mx / n, my / n, mz / n];
  }

  /// Approximate bounding radius from center.
  double get boundingRadius {
    final c = center;
    double maxR = 0;
    final n = vertexCount;
    for (int i = 0; i < n; i++) {
      final dx = vertices[i * 3] - c[0];
      final dy = vertices[i * 3 + 1] - c[1];
      final dz = vertices[i * 3 + 2] - c[2];
      final r = dx * dx + dy * dy + dz * dz;
      if (r > maxR) maxR = r;
    }
    return maxR > 0 ? (maxR * 1.0).sqrt() : 1.0;
  }
}

extension on double {
  double sqrt() {
    if (this <= 0) return 0;
    double x = this;
    double y = x / 2;
    for (int i = 0; i < 20; i++) {
      y = (y + x / y) / 2;
    }
    return y;
  }
}
