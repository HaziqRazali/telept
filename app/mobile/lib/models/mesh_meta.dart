/// Metadata from the server response (meta.json inside the ZIP).
class MeshMeta {
  final int frameCount;
  final double fps;

  const MeshMeta({required this.frameCount, required this.fps});

  factory MeshMeta.fromJson(Map<String, dynamic> json) {
    return MeshMeta(
      frameCount: (json['frame_count'] as num).toInt(),
      fps: (json['fps'] as num).toDouble(),
    );
  }
}
