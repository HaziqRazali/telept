import 'package:flutter/material.dart';
import 'config.dart';
import 'screens/home_screen.dart';
import 'services/mhr_model.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await AppConfig.init(); // load persisted server URL before first build
  await MhrModel.instance.load(); // pre-load MHR body model assets
  runApp(const TelePTApp());
}

class TelePTApp extends StatelessWidget {
  const TelePTApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TelePT Body Capture',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: Colors.indigo,
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      home: const HomeScreen(),
    );
  }
}
