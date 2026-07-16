import 'package:flutter/material.dart';
import 'aruco_tracker_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ArucoApp());
}

class ArucoApp extends StatelessWidget {
  const ArucoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ArUco Tracker',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const ArucoTrackerScreen(),
    );
  }
}
