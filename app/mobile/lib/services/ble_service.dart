import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:bluetooth_low_energy/bluetooth_low_energy.dart';
import 'package:flutter/material.dart' hide ConnectionState;
import 'package:permission_handler/permission_handler.dart';

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

  // Camera start/record can briefly delay BLE notifications on iOS.
  // Keep this generous so we don't force-disconnect a healthy link.
  static const int _idleDisconnectMs = 20000;

  final CentralManager _manager = CentralManager();

  final ValueNotifier<bool> isConnected = ValueNotifier(false);

  /// Notifier updated every time the discovered device list changes.
  final ValueNotifier<int> scanResultsCount = ValueNotifier(0);

  bool _working = false;
  bool _stopped = false;

  final List<DiscoveredEventArgs> _devList = [];
  final Map<BleDataSource, Peripheral> _dsPeripherals = {};
  final Set<BleDataSource> _activeSources = {};
  // Persists across disconnects – once we've seen a peripheral once we
  // can reconnect to it directly without waiting for another advertisement.
  final Map<BleDataSource, Peripheral> _knownPeripherals = {};
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
    _startLoop().whenComplete(() {
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

  // ── Source management ─────────────────────────────────────────────────────

  List<BleDataSource> get configuredSources => List.unmodifiable(_rdevs);

  bool isSourceConnected(BleDataSource ds) => _activeSources.contains(ds);

  Future<void> removeSource(BleDataSource ds) async {
    if (_dsPeripherals.containsKey(ds)) {
      try { await _manager.disconnect(_dsPeripherals[ds]!); } catch (_) {}
      _dsPeripherals.remove(ds);
    }
    _activeSources.remove(ds);
    _knownPeripherals.remove(ds);
    _rdevs.remove(ds);
    await AppConfig.setBleDevicesJson(BleDataSource.listToJson(_rdevs));
    _updateConnected();
  }

  Future<void> addSource(BleDataSource ds) async {
    if (_rdevs.any((r) => r.serviceID == ds.serviceID && r.charID == ds.charID)) return;
    _rdevs.add(ds);
    await AppConfig.setBleDevicesJson(BleDataSource.listToJson(_rdevs));
    // Restart loop so the new source is picked up
    if (!_working) {
      _working = true;
      _stopped = false;
      _startLoop().whenComplete(() => _working = false);
    }
  }

  Future<void> triggerScan() => _discoverDevices();

  // ── Internal ──────────────────────────────────────────────────────────────

  Future<void> _discoverDevices() async {
    _devList.clear();
    scanResultsCount.value = 0;

    if (Platform.isAndroid || Platform.isIOS) {
      await Permission.bluetooth.request();
    }

    try { await _manager.stopDiscovery(); } catch (_) {}

    // Retry startDiscovery up to 10 times, stopping once devices are found
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      for (int i = 0; i < 10 && _devList.isEmpty; i++) {
        try {
          await _manager.startDiscovery();
          return; // scan runs continuously until stop() or next _discoverDevices call
        } catch (_) {}
        await Future.delayed(const Duration(seconds: 2));
      }
    });
  }

  Future<void> _startLoop() async {
    _rdevs = BleDataSource.listFromJson(AppConfig.bleDevicesJson);

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
        scanResultsCount.value = _devList.length;

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
      if (dargs.state == ConnectionState.disconnected) {
        _dropPeripheral(dargs.peripheral);
        // Immediately re-populate _dsPeripherals from the known map so the
        // connect block retries GATT on the very next loop tick without
        // waiting for a new scan advertisement.
        for (final ds in _rdevs) {
          if (!_dsPeripherals.containsKey(ds) &&
              _knownPeripherals.containsKey(ds)) {
            _dsPeripherals[ds] = _knownPeripherals[ds]!;
          }
        }
      }
      _updateConnected();
    });

    await _discoverDevices();

    while (!_stopped) {
      // Rescan only when devList is empty and not all sources are connected
      if (_devList.isEmpty &&
          !_rdevs.every((ds) => _dsPeripherals.containsKey(ds))) {
        await _discoverDevices();
        await Future.delayed(const Duration(seconds: 2));
        if (_devList.isEmpty) continue;
      }

      final tnow = DateTime.now().millisecondsSinceEpoch;
      for (final dt in Map.from(mapDevTimes).entries) {
        // Do not force-disconnect while recording sensor data.
        if (!_capturing && tnow - dt.value.ts > _idleDisconnectMs) {
          _dropPeripheral(dt.key);
          try {
            await _manager.disconnect(dt.key);
          } catch (_) {}
          mapDevTimes.remove(dt.key);
          _updateConnected();
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
              final svcMatches = services
                  .where((s) => s.uuid.toString() == ent.key.serviceID);
              if (svcMatches.isEmpty) {
                _dropPeripheral(device);
                break;
              }

              final charMatches = svcMatches.first.characteristics
                  .where((c) => c.uuid.toString() == ent.key.charID);
              if (charMatches.isEmpty) {
                _dropPeripheral(device);
                break;
              }

              await _manager.setCharacteristicNotifyState(
                  device, charMatches.first,
                  state: true);
            } catch (e) {
              _dropPeripheral(device);
              try { await _manager.disconnect(device); } catch (_) {}
              break;
            }
            _knownPeripherals[ent.key] = device; // remember for instant reconnect
            _activeSources.add(ent.key);
            mapDevTimes[device] = _DeviceStats();
            _updateConnected();
            break;
          }
        }
      }

      await Future.delayed(const Duration(seconds: 2));
    }

    await _manager.stopDiscovery();
    await _scanSub?.cancel();
    for (final p in _dsPeripherals.values) {
      try { await _manager.disconnect(p); } catch (_) {}
    }
    _activeSources.clear();
    _dsPeripherals.clear();
    _knownPeripherals.clear();
    _devList.clear();
    scanResultsCount.value = 0;
    await _stateSub?.cancel();
    _updateConnected();
  }

  void _updateConnected() {
    isConnected.value = _activeSources.isNotEmpty;
  }

  void _dropPeripheral(Peripheral peripheral) {
    final matches = _dsPeripherals.entries
        .where((ent) => ent.value == peripheral)
        .map((ent) => ent.key)
        .toList();
    for (final ds in matches) {
      _dsPeripherals.remove(ds);
      _activeSources.remove(ds);
    }
  }

  // ── BLE setup dialog (mirrors icam's showBLEDialog) ──────────────────────

  /// Opens the BLE management dialog: shows configured sources, lets user
  /// delete them or add new ones by scanning.
  Future<void> showBleDialog(BuildContext ctx) async {
    if (!_working) {
      _working = true;
      _stopped = false;
      _startLoop().whenComplete(() => _working = false);
    }

    // ── Step 1: show currently configured sources ─────────────────────
    if (_rdevs.isNotEmpty && ctx.mounted) {
      final addNew = await showDialog<bool>(
        context: ctx,
        builder: (context) => StatefulBuilder(
          builder: (ctx2, setState) => AlertDialog(
            title: const Text('BLE Sensor Sources'),
            content: SizedBox(
              width: MediaQuery.of(context).size.width * 0.8,
              child: ListView.builder(
                shrinkWrap: true,
                itemCount: _rdevs.length,
                itemBuilder: (_, i) {
                  final ds = _rdevs[i];
                  final connected = _activeSources.contains(ds);
                  return Card(
                    margin: const EdgeInsets.symmetric(vertical: 4),
                    child: ListTile(
                      leading: Icon(
                        Icons.bluetooth,
                        color: connected ? Colors.greenAccent : Colors.grey,
                      ),
                      title: Text(ds.name),
                      subtitle: Text(
                        'Service: ${ds.serviceID}\nChar: ${ds.charID}',
                        style: const TextStyle(fontSize: 11),
                      ),
                      trailing: IconButton(
                        icon: const Icon(Icons.delete, color: Colors.red),
                        onPressed: () async {
                          await removeSource(ds);
                          setState(() {});
                        },
                      ),
                    ),
                  );
                },
              ),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx2, false),
                child: const Text('Close'),
              ),
              FilledButton(
                onPressed: () => Navigator.pop(ctx2, true),
                child: const Text('Add device'),
              ),
            ],
          ),
        ),
      );
      if (addNew != true) return;
    }

    if (!ctx.mounted) return;

    // ── Step 2: show discovered devices (live-updating) ───────────────
    _devList.clear();
    scanResultsCount.value = 0;
    unawaited(_discoverDevices());

    final selectedIndex = await showDialog<int>(
      context: ctx,
      builder: (context) => ValueListenableBuilder<int>(
        valueListenable: scanResultsCount,
        builder: (_, __, ___) => AlertDialog(
          title: Text('Scanning... (${_devList.length} found)'),
          content: SizedBox(
            width: MediaQuery.of(context).size.width * 0.8,
            child: _devList.isEmpty
                ? const SizedBox(
                    height: 80,
                    child: Center(child: CircularProgressIndicator()),
                  )
                : ListView.builder(
                    shrinkWrap: true,
                    itemCount: _devList.length,
                    itemBuilder: (_, i) {
                      final dev = _devList[i];
                      return ListTile(
                        leading: const Icon(Icons.bluetooth_searching),
                        title: Text(dev.advertisement.name ?? 'Unknown'),
                        subtitle: Text(dev.peripheral.uuid.toString()),
                        onTap: () => Navigator.of(context).pop(i),
                      );
                    },
                  ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                _devList.clear();
                scanResultsCount.value = 0;
                unawaited(_discoverDevices());
              },
              child: const Text('Re-scan'),
            ),
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
          ],
        ),
      ),
    );

    if (selectedIndex == null || !ctx.mounted) return;

    final selectedPeripheral = _devList[selectedIndex].peripheral;

    // ── Step 3: connect and discover GATT ────────────────────────────
    if (ctx.mounted) {
      showDialog(
        context: ctx,
        barrierDismissible: false,
        builder: (_) => const Center(child: CircularProgressIndicator()),
      );
    }

    Object? connectErr;
    List<GATTService> services = [];
    try {
      try { await _manager.connect(selectedPeripheral); } catch (_) {}
      services = await _manager.discoverGATT(selectedPeripheral);
    } catch (e) {
      connectErr = e;
    } finally {
      if (ctx.mounted && Navigator.canPop(ctx)) Navigator.of(ctx).pop();
    }

    if (connectErr != null || services.isEmpty || !ctx.mounted) return;

    // ── Step 4: pick a service ────────────────────────────────────────
    final serviceIndex = await showDialog<int>(
      context: ctx,
      builder: (context) => AlertDialog(
        title: const Text('Select service'),
        content: SizedBox(
          width: MediaQuery.of(context).size.width * 0.8,
          child: ListView.builder(
            shrinkWrap: true,
            itemCount: services.length,
            itemBuilder: (_, i) => ListTile(
              title: Text(services[i].uuid.toString()),
              onTap: () => Navigator.pop(context, i),
            ),
          ),
        ),
        actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel'))],
      ),
    );
    if (serviceIndex == null || !ctx.mounted) return;

    final selectedService = services[serviceIndex];
    final chars = selectedService.characteristics;

    // ── Step 5: pick a characteristic ────────────────────────────────
    final charIndex = await showDialog<int>(
      context: ctx,
      builder: (context) => AlertDialog(
        title: const Text('Select characteristic'),
        content: SizedBox(
          width: MediaQuery.of(context).size.width * 0.8,
          child: ListView.builder(
            shrinkWrap: true,
            itemCount: chars.length,
            itemBuilder: (_, i) => ListTile(
              title: Text(chars[i].uuid.toString()),
              onTap: () => Navigator.pop(context, i),
            ),
          ),
        ),
        actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel'))],
      ),
    );
    if (charIndex == null || !ctx.mounted) return;

    // ── Step 6: give it a name and save ──────────────────────────────
    final nameController = TextEditingController(
        text: _devList[selectedIndex].advertisement.name ?? 'Sensor');
    final confirmed = await showDialog<bool>(
      context: ctx,
      builder: (context) => AlertDialog(
        title: const Text('Name this sensor'),
        content: TextField(
          controller: nameController,
          decoration: const InputDecoration(
            labelText: 'Name',
            border: OutlineInputBorder(),
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Save')),
        ],
      ),
    );
    if (confirmed != true) return;

    await addSource(BleDataSource(
      name: nameController.text.trim().isEmpty
          ? (_devList[selectedIndex].advertisement.name ?? 'Sensor')
          : nameController.text.trim(),
      serviceID: selectedService.uuid.toString(),
      charID: chars[charIndex].uuid.toString(),
    ));

    if (ctx.mounted) {
      ScaffoldMessenger.of(ctx).showSnackBar(
        SnackBar(content: Text('Sensor "${nameController.text.trim()}" saved.')),
      );
    }
  }
}
