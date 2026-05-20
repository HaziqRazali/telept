import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:aruco_tracker/main.dart';

void main() {
  testWidgets('App smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const ArucoApp());
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
