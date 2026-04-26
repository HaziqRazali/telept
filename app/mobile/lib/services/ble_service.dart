import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:bluetooth_low_energy/bluetooth_low_energy.dart';
import 'package:flutter/material.dart' hide ConnectionState;

import '../config.dart';
import '../models/sensor_sample.dart';

// ---------------------------------------------------------------------------
// BleDataSource  –  mirrors icam's BleDataSource exactly
// ---------------------------------------------------------------------------

class BleDataSource {
  final String name;
  final String serviceID;
  final String charID;

  const BleDataSource({
    required this.name,
    required this.serviceID,
    required this.charID,
  });

  Map<String, dynamic> toMap() => {
        'name': name,
        'serviceID': serviceID,
        'charID': charID,
      };

  factory BleDataSource.fromMap(Map<String, dynamic> map) => BleDataSource(
        name: map['name'] as String? ?? '',
        serviceID: map['serviceID'] as String? ?? '',
        charID: map['charID'] as String? ?? '',
      );

  static String listToJson(List<BleDataSource> list) =>
      json.encode(list.map((e) => e.toMap()).toList());

  static List<BleDataSource> listFromJson(String? source) {
    if (source == null || source.isEmpty) return [];
    try {
      final decoded = json.decode(source) as List<dynamic>;
      return decoded.map((e) => BleDataSource.fromMap(e as Map<String, dynamic>)).toList();
    } on Exception {
      return [];
    }
  }
}

// ---------------------------------------------------------------------------
// _DeviceStats
// ---------------------------------------------------------------------------

class _DeviceStats {
  int rc = 0;
  int ts = DateTime.now().millisecondsSinceEpoch;
}

// ---------------------------------------------------------------------------
// BleService  –  global singleton
// ---------------------------------------------------------------------------

final bleService = BleService._();

class BleService {
  BleService._();

  final CentralManager _manager = CentralManager();

  final ValueNotifier<bool> isConnected = ValueNotifier(false);

  bool _working = false;
  bool _stopped = false;

  final List<DiscoveredEventArgs> _devList = [];
  final Map<BleDataSource, Peripheral> _dsPeripherals = {};
  List<BleDataSource> _rdevs = [];

  StreamSubscription<DiscoveredEventArgs>? _scanSub;
  StreamSubscription<BluetoothLowEnergyStateChangedEventArgs>? _stateSub;

  // Capture state
  bool _capturing = false;
  int _captureT0 = 0;
  final List<SensorSample> _captureBuffer = [];

  // ── Public API ────────────────────────────────────────────────────────────

  /// Start BLE scanning and connection loop. Call once from HomeScreen.initState.
  void start(BuildContext ctx) {
    if (_working) return;
    _working = true;
    _stopped = false;
    _startLoop(ctx).whenComplete(() {
      _working = false;
    });
  }

  /// Stop all BLE activity.
  void stop() {
    _stopped = true;
  }

  /// Begin collecting sensor samples. [t0Ms] is the recording start epoch ms.
  void startCapture(int t0Ms) {
    _captureBuffer.clear();
    _captureT0 = t0Ms;
    _capturing = true;
  }

  /// Stop collecting and return the captured samples (null if nothing captured).
  List<SensorSample>? stopCapture() {
    _capturing = false;
    if (_captureBuffer.isEmpty) return null;
    final result = List<SensorSample>.unmodifiable(_captureBuffer);
    _captureBuffer.clear();
    return result;
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  Future<void> _discoverDevices() async {
    try {
      await _manager.startDiscovery();
      await Future.delayed(const Duration(seconds: 4));
      await _manager.stopDiscovery();
    } catch (_) {}
  }

  Future<void> _startLoop(BuildContext ctx) async {
    _rdevs = BleDataSource.listFromJson(AppConfig.bleDevicesJson);
    if (_rdevs.isEmpty) return;

    final Map<Peripheral, _DeviceStats> mapDevTimes = {};
    int rt0 = DateTime.now().millisecondsSinceEpoch;

    _stateSub = _manager.stateChanged.listen((args) async {
      if (args.state == BluetoothLowEnergyState.unauthorized &&
          Platform.isAndroid) {
        await _manager.authorize();
      }
    });

    _scanSub = _manager.discovered.listen((args) {
      if (args.advertisement.serviceUUIDs.isNotEmpty) {
        final index = _devList.indexWhere((d) => d.peripheral == args.peripheral);
        if (index < 0) {
          _devList.add(args);
        } else {
          _devList[index] = args;
        }

        for (final ds in _rdevs) {
          if (_dsPeripherals[ds] == null &&
              args.advertisement.serviceUUIDs
                  .any((uuid) => uuid.toString() == ds.serviceID)) {
            _dsPeripherals[ds] = args.peripheral;
          }
        }
      }
    });

    _manager.characteristicNotified.listen((cargs) {
      final ts = DateTime.now().millisecondsSinceEpoch;

      for (final ent in Map.from(_dsPeripherals).entries) {
        if (cargs.peripheral == ent.value &&
            cargs.characteristic.uuid.toString() == ent.key.charID) {
          final ds = mapDevTimes[ent.value] ?? _DeviceStats();
          mapDevTimes[ent.value] = ds;
          ds.ts = ts;
          ds.rc++;

          final rawString = utf8.decode(cargs.value);
          final parts = rawString.split(',');

          if (parts.length >= 2) {
            try {
              final direction = double.parse(parts[0].trim());
              final force = (double.parse(parts[1].trim()) + 203658) / 68551;

              if (_capturing) {
                _captureBuffer.add(SensorSample(
                  timestampMs: ts - _captureT0,
                  force: force,
                  direction: direction,
                ));
              }

              if (ts - rt0 >= 1000) {
                ds.rc = 0;
                rt0 = ts;
              }
            } catch (_) {}
          }
        }
      }
    });

    _manager.connectionStateChanged.listen((dargs) {
      for (final ent in Map.from(_dsPeripherals).entries) {
        if (dargs.peripheral == ent.value) {
          if (dargs.state == ConnectionState.disconnected) {
            _dsPeripherals.remove(ent.key);
          }
        }
      }
      _updateConnected();
    });

    await _discoverDevices();

    while (!_stopped) {
      if (_devList.isEmpty &&
          !_rdevs.every((ds) => _dsPeripherals.containsKey(ds))) {
        await _discoverDevices();
        await Future.delayed(const Duration(seconds: 2));
        if (_devList.isEmpty) continue;
      }

      final tnow = DateTime.now().millisecondsSinceEpoch;
      for (final dt in Map.from(mapDevTimes).entries) {
        if (tnow - dt.value.ts > 2000) {
          try {
            await _manager.disconnect(dt.key);
          } catch (_) {}
          mapDevTimes.remove(dt.key);
        }
      }

      for (final ent in Map.from(_dsPeripherals).entries) {
        if (!mapDevTimes.containsKey(ent.value)) {
          final device = ent.value;
          while (true) {
            try {
              try {
                await _manager.connect(device);
              } catch (_) {}

              final services = await _manager.discoverGATT(device);
              final svcMatches =
                  services.where((s) => s.uuid.toString() == ent.key.serviceID);
              if (svcMatches.isEmpty) break;

              final charMatches = svcMatches.first.characteristics
                  .where((c) => c.uuid.toString() == ent.key.charID);
              if (charMatches.isEmpty) break;

              await _manager.setCharacteristicNotifyState(
                  device, charMatches.first,
                  state: true);

              mapDevTimes[device] = _DeviceStats();
              _updateConnected();
              break;
            } catch (_) {
              break;
            }
          }
        }
      }

      await Future.delayed(const Duration(seconds: 1));
    }

    await _scanSub?.cancel();
    await _stateSub?.cancel();
    _dsPeripherals.clear();
    _devList.clear();
    _updateConnected();
  }

  void _updateConnected() {
    isConnected.value = _dsPeripherals.isNotEmpty;
  }
}
