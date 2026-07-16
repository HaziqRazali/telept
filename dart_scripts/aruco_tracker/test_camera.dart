import 'package:opencv_dart/opencv_dart.dart' as cv;

void main() {
  print('Opening camera...');
  final cap = cv.VideoCapture.fromDevice(0);
  print('isOpened: ${cap.isOpened}');

  if (!cap.isOpened) {
    print('ERROR: Camera not opened');
    return;
  }

  // Same set() calls as the Flutter app
  cap.set(cv.CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480);
  cap.set(cv.CAP_PROP_FPS, 30);
  print('after set: W=${cap.get(cv.CAP_PROP_FRAME_WIDTH)} H=${cap.get(cv.CAP_PROP_FRAME_HEIGHT)}');

  // Warm-up (same as app)
  for (var i = 0; i < 10; i++) {
    cap.read();
  }

  // Actual reads
  for (var i = 0; i < 5; i++) {
    final (grabbed, frame) = cap.read();
    print('read $i: grabbed=$grabbed  empty=${frame.isEmpty}  '
        'size=${frame.cols}x${frame.rows}');
    frame.dispose();
  }

  cap.release();
  print('Done');
}
