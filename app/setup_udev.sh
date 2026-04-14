#!/bin/bash
# Run this once as sudo to allow Linux to see Samsung Android devices over USB.
# Usage:  sudo bash /data/telept/app/setup_udev.sh

set -e

echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="04e8", MODE="0666", GROUP="plugdev"' \
    > /etc/udev/rules.d/51-android.rules

chmod a+r /etc/udev/rules.d/51-android.rules
udevadm control --reload-rules
udevadm trigger

echo "Done. Now unplug and re-plug your Samsung phone, then run:"
echo "  /home/haziq/android-sdk/platform-tools/adb devices"
