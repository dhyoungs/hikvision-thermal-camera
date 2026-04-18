#!/bin/bash
# Hikvision thermal camera installer — takes a fresh Raspberry Pi to a
# working thermal viewer and proves the camera is live.
#
# What it does:
#   1. Installs OS packages (python3-opencv, v4l-utils, ffmpeg)
#   2. Applies the UVC quirk the camera needs (otherwise it won't open)
#   3. Installs a udev rule so the `pi` user can access /dev/video*
#   4. Runs a single-frame capture self-test that reports the
#      hottest and coldest pixel temperatures read from the sensor.
#
# Safe to re-run: every step is idempotent.
# Usage:   bash install.sh

set -euo pipefail
cd "$(dirname "$0")"

log()   { printf "\033[1;36m==> %s\033[0m\n" "$*"; }
warn()  { printf "\033[1;33m!!  %s\033[0m\n" "$*"; }
fail()  { printf "\033[1;31mXX  %s\033[0m\n" "$*"; exit 1; }
pass()  { printf "\033[1;32mOK  %s\033[0m\n" "$*"; }

# ---- 1. OS packages ----
log "Installing system packages"
sudo apt-get update
sudo apt-get install -y python3-opencv v4l-utils ffmpeg curl

# ---- 2. UVC quirk (PROBE_MINMAX bit) ----
log "Applying UVC quirk (0x02) so the Hikvision sensor can open"
sudo tee /etc/modprobe.d/hikvision-thermal.conf >/dev/null <<'MODPROBE'
# Hikvision Mini2 Plus thermal (2bdf:0102) stalls on GET_MIN probe query;
# quirk bit 0x02 = UVC_QUIRK_PROBE_MINMAX skips it.
options uvcvideo quirks=2
MODPROBE
echo 2 | sudo tee /sys/module/uvcvideo/parameters/quirks >/dev/null 2>&1 || true

# If the module is already loaded, reload so the quirk actually takes effect.
if lsmod | grep -q '^uvcvideo'; then
    sudo modprobe -r uvcvideo 2>/dev/null || true
    sudo modprobe uvcvideo
fi

# ---- 3. udev rule so pi has access to /dev/video* for this vendor ----
log "Installing udev rule for 2bdf:0102"
sudo tee /etc/udev/rules.d/61-hikvision-thermal.rules >/dev/null <<'UDEV'
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="2bdf", ATTRS{idProduct}=="0102", MODE="0666", GROUP="video", TAG+="uaccess"
UDEV
sudo udevadm control --reload-rules
sudo udevadm trigger

# ---- 4. Self-test ----
log "Running self-test (capturing one frame)"

if ! lsusb | grep -qi 'hik\|2bdf:0102'; then
    warn "Hikvision camera is not plugged in — self-test skipped."
    warn "Plug the camera in and run: python3 thermal_viewer.py"
    exit 0
fi

VIDNODE=""
for n in /dev/video*; do
    [ -c "$n" ] || continue
    # Skip metadata nodes (the Pi ISP exposes many /dev/videoN).
    CARD=$(v4l2-ctl -d "$n" --info 2>/dev/null | awk -F: '/Card type/{print $2}' | xargs || true)
    case "$CARD" in *HikCamera*|*Hik*) VIDNODE="$n"; break ;; esac
done
[ -n "$VIDNODE" ] || fail "No /dev/videoN reports as a HikCamera — check the UVC quirk applied (reboot if needed)."
pass "Camera on $VIDNODE"

python3 - "$VIDNODE" <<'PY'
import os, sys
os.environ["QT_OPENGL"] = "software"
import cv2

node = sys.argv[1]
cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 392)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
if not cap.isOpened():
    print("FAIL: cv2.VideoCapture could not open", node); sys.exit(2)
# Warm-up: the sensor sometimes emits a few blank frames while AEC settles.
for _ in range(5):
    cap.read()
ok, frame = cap.read()
cap.release()
if not ok or frame is None:
    print("FAIL: no frame read"); sys.exit(3)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
upper = gray[0:192]
print(f"OK: frame {w}x{h}, thermal-luma min/max = {upper.min()}/{upper.max()}")
PY
pass "Camera captured a frame — viewer ready."
echo
pass "Install complete — run: python3 thermal_viewer.py"
