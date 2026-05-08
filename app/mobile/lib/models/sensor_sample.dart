/// A single BLE sensor reading captured during video recording.
class SensorSample {
  /// Milliseconds elapsed since the recording started (t0).
  final int timestampMs;

  /// Load-cell force value (calibrated, kg).
  final double force;

  /// Direction value (degrees).
  final double direction;

  const SensorSample({
    required this.timestampMs,
    required this.force,
    required this.direction,
  });
}
