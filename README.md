# Hikvision USB Thermal Camera Viewer

Real-time thermal viewer and target tracker for the **Hikvision Mini2 Plus**
USB thermal camera (`2bdf:0102`), running on a Raspberry Pi 5.

Displays live thermal video with temperature readout, colour palette,
neural-network target tracking, and recording / replay tools.

## Quick start

```bash
cd ~
git clone https://github.com/dhyoungs/hikvision-thermal-camera.git
cd hikvision-thermal-camera
bash install.sh
python3 thermal_viewer.py
```

The install script applies the UVC quirk (without which the camera won't
open), installs the system OpenCV, sets up the udev rule, and runs a
self-test that captures one frame and reports min/max pixel temperatures.

## Docs

| File | Who it's for |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Future Claude sessions — full protocol/hardware notes |
| [thermal_camera_notes.md](thermal_camera_notes.md) | Developer implementation reference |
| [docs/USER_GUIDE.docx](docs/USER_GUIDE.docx) | Human-readable operator manual |

## Requires

- Raspberry Pi 5 running Raspberry Pi OS (64-bit, Bookworm or newer)
- Hikvision Mini2 Plus thermal USB camera
- `python3-opencv` (system package — the pip wheel's V4L2 backend is broken on aarch64)
