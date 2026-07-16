import 'dart:io';

import 'package:flutter/material.dart';

import 'screens/camera_screen.dart';
import 'services/mhr_model.dart';
import 'services/pose_estimator_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // On Linux, kpts2smpl inference runs inside the Python bridge — skip TFLite load.
  final futures = <Future>[MhrModel.instance.load()];
  if (!Platform.isLinux) {
    futures.add(PoseEstimatorService.instance.load());
  }
  await Future.wait(futures);
  runApp(const PoseEstimatorApp());
}

class PoseEstimatorApp extends StatelessWidget {
  const PoseEstimatorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pose Estimator',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true),
      home: const CameraScreen(),
    );
  }
}
