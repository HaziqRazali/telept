#include "my_application.h"

#include <stdlib.h>  // setenv

int main(int argc, char** argv) {
  // Set OpenCV V4L2 select() timeout to 0 so that when VIDIOC_DQBUF returns
  // EAGAIN (O_NONBLOCK device), the fallback select() check returns immediately
  // instead of blocking for 10 s.  Blocking select() is interrupted every ~1 ms
  // in debug mode by the Dart VM's SIGPROF CPU-profiler, causing grab() to
  // always return false (EINTR path in tryIoctl).  With timeout=0 the select()
  // is a non-blocking poll, so SIGPROF never has a chance to interrupt it.
  setenv("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "0", 0);

  g_autoptr(MyApplication) app = my_application_new();
  return g_application_run(G_APPLICATION(app), argc, argv);
}
