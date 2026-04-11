# Hikvision USB Thermal Camera — Complete Implementation Notes

This document is the authoritative reference for recreating `/home/pi/thermal_viewer.py`
from scratch. It covers every hardware quirk, data-format detail, and design decision
discovered during development.

---

## Hardware

| Item | Value |
|------|-------|
| Device | HIK HikCamera |
| USB ID | `2bdf:0102` |
| Serial | F10615613 |
| Platform | Raspberry Pi 5, aarch64, Linux 6.12, Raspberry Pi OS Bookworm |
| Video node | `/dev/video0` (thermal) |

---

## Step 1 — UVC quirk (required or camera won't open)

The camera stalls when the UVC driver sends a `GET_MIN` probe control query (`0x82`),
causing `EPIPE` and making all subsequent ioctls return `EIO`.

**Fix:** Set UVC quirk bit `0x02` (`UVC_QUIRK_PROBE_MINMAX`).

Apply at runtime:
```bash
echo 2 | sudo tee /sys/module/uvcvideo/parameters/quirks
```

Make permanent — create `/etc/modprobe.d/hikvision-thermal.conf`:
```
options uvcvideo quirks=2
```

Reload: `sudo modprobe -r uvcvideo && sudo modprobe uvcvideo` (or reboot).

---

## Step 2 — OpenCV package

The pip `opencv-python` wheel (manylinux) has a non-functional V4L2 backend on aarch64.
Use the system package:

```bash
pip3 uninstall opencv-python
sudo apt-get install python3-opencv   # 4.10.x on Bookworm
```

---

## Step 3 — Qt blank window fix

Qt5 OpenCV on Pi uses OpenGL by default → blank window on Pi 5 framebuffer.

```python
import os
os.environ.setdefault("QT_OPENGL", "software")
import cv2
```

This **must** be set before `import cv2`. Also avoid non-ASCII characters in window titles.

---

## Step 4 — Frame format: 256×392 YUYV

OpenCV's V4L2 backend can only negotiate `256×192` (visual only, no temperature).
Opening at `256×392` via raw V4L2 ioctls yields the full data frame:

| Rows | Content |
|------|---------|
| 0 – 191 | Visual thermal image (192 rows × 256 pixels, YUYV) |
| 192 | Header row — image dimensions `(0x00, 0xC0=192, 0x01, 0x00=256)` |
| 193 | Magic sync row — `0xAA 0xBB 0xCC 0xDD` repeated (stored LE: `dd cc bb aa`) |
| 194 – 195 | Unused / padding |
| **196 – 387** | **Per-pixel temperature (192 rows × 256 values)** |
| 388 | Calibration footer |
| 389 – 391 | Unused / padding |

### Temperature encoding

Each row is 512 bytes (2 bytes/pixel × 256 pixels). Temperature for pixel `(row, col)`:

```
u16 = (byte[2*col] << 8) | byte[2*col + 1]   # big-endian uint16
T(°C) = u16 / 100.0 - 273.15
```

**Critical:** divisor is **100**, not 64. Confirmed empirically — BE/divisor=100 gives
`valid_frac=0.31, mean=26.2°C` for a room-temperature scene. With divisor=64 the mean
is ~176°C (wrong).

The magic row (193) byte order is little-endian, but the temperature uint16 within
YUYV byte-pairs is big-endian. This asymmetry is consistent — the camera packs raw
uint16 values into the YUYV stream regardless of YUYV channel semantics.

### Black-hot polarity

The camera streams in **black-hot** mode: cold objects are bright, hot objects are dark
in the raw Y channel. Must invert before display and blob detection:

```python
visual = (255 - frame[:192, 0::2]).astype(np.uint8)
```

### Border pixel trap

~69% of the 256×192 grid consists of masked/dead border pixels with raw Y ≈ 14–26.
After inversion these become 229–241 — the **brightest** values in the image.
**Never use the visual Y channel for blob/hot-region detection** — always use the
temperature array (where border pixels decode to T ≈ −220°C and are NaN-masked).

```python
T[(T < -40) | (T > 400)] = np.nan
```

---

## Step 5 — Raw V4L2 capture (ctypes/libc, aarch64)

The `v4l2_buffer` struct is **88 bytes** on aarch64 (not 68 as on x86).

```python
_libc = ctypes.CDLL("libc.so.6", use_errno=True)

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP            = 1
V4L2_PIX_FMT_YUYV           = 0x56595559

# ioctl codes — direction=3 (RW), type='V'=86, size=88
VIDIOC_QUERYBUF = (3<<30)|(ord('V')<<8)|9 |(88<<16)
VIDIOC_QBUF     = (3<<30)|(ord('V')<<8)|15|(88<<16)
VIDIOC_DQBUF    = (3<<30)|(ord('V')<<8)|17|(88<<16)
# These two are size-independent:
VIDIOC_S_FMT    = 0xC0D05605
VIDIOC_REQBUFS  = 0xC0145608
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF= 0x40045613

# v4l2_buffer field offsets (88-byte struct, aarch64)
_BUF_OFF_INDEX  = 0
_BUF_OFF_TYPE   = 4
_BUF_OFF_USED   = 8    # bytesused
_BUF_OFF_MEM    = 60   # memory type
_BUF_OFF_MOFF   = 64   # m.offset (mmap offset)
_BUF_OFF_LEN    = 72   # length
```

Capture sequence:
1. `VIDIOC_S_FMT` — request `256×392 YUYV`
2. `VIDIOC_REQBUFS` — allocate N mmap buffers
3. `VIDIOC_QUERYBUF` + `mmap()` + `VIDIOC_QBUF` for each buffer
4. `VIDIOC_STREAMON`
5. Loop: `select()` → `VIDIOC_DQBUF` → copy → `VIDIOC_QBUF`
6. `VIDIOC_STREAMOFF` on exit

`select()` is required — fd is opened `O_NONBLOCK` so DQBUF returns `EAGAIN` otherwise.

### `ThermalCamera.read()` returns three values

```python
visual, T, u16_raw = cam.read(timeout=1.0)
# visual   : uint8  (192, 256)   Y-channel, black-hot inverted
# T        : float32(192, 256)   °C, NaN where invalid
# u16_raw  : uint16 (192, 256)   raw encoded values (for calibration)
```

---

## Step 6 — Extracting visual and temperature

```python
frame = raw_bytes.reshape(392, 512)   # 256 YUYV pixels × 2 bytes = 512 bytes/row

# Visual: Y channel, inverted (black-hot → white-hot)
visual = (255 - frame[:192, 0::2]).astype(np.uint8)

# Temperature
tr  = frame[196:388].reshape(192, 256, 2)
u16 = (tr[:,:,0].astype(np.uint16) << 8) | tr[:,:,1].astype(np.uint16)
T   = u16.astype(np.float32) / 100.0 - 273.15
T[(T < -40) | (T > 400)] = np.nan
```

---

## Step 7 — Display: greyscale + colour for top 10% hottest pixels

Display is greyscale everywhere except the top `(100 − HOT_DISPLAY_PCT)`% of valid
pixels, which are shown in the Inferno colormap. This is computed dynamically per frame
from the temperature array (not a fixed threshold).

```python
HOT_DISPLAY_PCT = 90   # show colour for top 10% of valid-T pixels

T_valid = T[~np.isnan(T)]
hot_thresh = float(np.percentile(T_valid, HOT_DISPLAY_PCT)) if len(T_valid) else 999.0
hot_mask   = ((~np.isnan(T)) & (T >= hot_thresh)).astype(np.uint8)

gray_bgr = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
colored  = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
combined = np.where(hot_mask[:,:,np.newaxis], colored, gray_bgr)
thermal  = cv2.resize(combined, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
```

Use `INTER_LINEAR` (not INTER_NEAREST) to avoid blocky appearance at 3× upscale.

---

## Step 8 — Hot blob tracker (`HotBlobTracker`)

### Two-state design

**SEARCHING**: each frame find the globally hottest valid blob. Must be stable
(centroid drift < 20 sensor px between frames) for ≥ 0.25 s → acquires lock.

**LOCKED**: predicts next centroid using EMA-smoothed velocity. Searches within an
expanding radius around the prediction. Holds lock through occlusions up to 1.5 s.
Only breaks lock if object truly disappears.

### Key constants

```python
CONFIRM_SECS   = 0.25   # stability window to acquire lock
MAX_DRIFT_PX   = 20     # max per-frame jump during SEARCHING
SEARCH_RADIUS  = 35     # base search radius (sensor px) when LOCKED
VEL_ALPHA      = 0.65   # velocity EMA (higher = slower to change)
MAX_MISS_SECS  = 1.5    # coast this long without detection before dropping lock
MIN_TEMP_RATIO = 0.50   # locked candidate must be ≥ 50% of frame max temperature
```

### Blob finding — MUST use temperature, not visual Y

```python
def _find_all_blobs(self, T):
    valid = ~np.isnan(T)
    thresh_T = float(np.percentile(T[valid], BLOB_THRESHOLD_PCT))  # e.g. 90th pct
    hot_mask = (valid & (T >= thresh_T)).astype(np.uint8) * 255
    # morphological open to remove single-pixel noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_OPEN, kernel)
    # connectedComponents → filter area > BLOB_MIN_AREA (9)
    ...
```

The visual Y channel **cannot** be used here — border pixels are always the brightest
after inversion and would win every frame.

### Scoring candidates when LOCKED

```python
score = blob["t_mean"] - distance_to_prediction * 0.5
```

Candidate must also be within `radius` and have `t_mean >= frame_max_T * 0.5`.

---

## Step 9 — Coordinate convention

Image coordinates have Y increasing downward. Displayed offsets use:
- **X**: positive = right of image centre
- **Y**: positive = above image centre (inverted)

```python
dx =   blob_cx - THERMAL_W // 2
dy = -(blob_cy - THERMAL_H // 2)
# Displayed as: X:+14 Y:-7
```

---

## Step 10 — Temperature calibration

### Formula

```python
T_raw = u16 / 100.0 - 273.15           # baseline
T_cal = T_raw * _cal_slope + _cal_offset   # after calibration
```

### Key targets
- `_cal_temp_hot = 80.0` (kettle reference, **K key**)
- `_cal_temp_amb = 22.0` (ambient reference, **A key**)

### Key press behaviour
- **K**: captures 99th-percentile u16 of the entire frame as hot reference. No aiming needed — fill frame with the hot object.
- **A**: captures median u16 of the entire frame as ambient reference.
- **0**: resets both references, slope=1, offset=0.

### Single-point calibration (K only — most common)

Treats the known point as a true Kelvin value and rescales the Kelvin axis:

```python
raw_K  = _cal_raw_hot / 100.0          # raw Kelvin value from camera
true_K = _cal_temp_hot + 273.15        # true Kelvin
_cal_slope  = true_K / raw_K
_cal_offset = (_cal_slope - 1.0) * 273.15
```

This corrects both scale and zero-point with a single reference.

### Two-point calibration (K + A)

Full linear: `slope = (T_hot − T_amb) / (raw_hot − raw_amb)`, `offset = T_hot − slope * raw_hot`.

### Persistence

Calibration auto-saved to `~/.hikcam_cal.json` after each K/A/0 press.
Auto-loaded at startup. JSON fields: `raw_hot, raw_amb, temp_hot, temp_amb, slope, offset`.

---

## Step 11 — Video recording

Press **R** to start recording; press **R** again to stop.

Two files are written simultaneously per recording session:
- `thermal_raw_YYYYMMDD_HHMMSS.avi` — frame before any HUD or blob overlay (pure display image)
- `thermal_ann_YYYYMMDD_HHMMSS.avi` — fully annotated frame with HUD, blob box, temperatures

Both use MJPG codec at 25 fps, frame size = `(THERMAL_W * SCALE, THERMAL_H * SCALE)` = `(768, 576)`.

```python
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
raw_w  = cv2.VideoWriter(f"thermal_raw_{ts}.avi", fourcc, 25.0, (disp_w, disp_h))
ann_w  = cv2.VideoWriter(f"thermal_ann_{ts}.avi", fourcc, 25.0, (disp_w, disp_h))
```

`raw_frame = thermal.copy()` must be taken **before** any draw calls on `canvas`.
Writers are written to after `cv2.imshow()` each frame.
Writers are released on stop (R) or on quit (Q), whichever comes first.

A pulsing red dot + `REC Ns` counter is shown in the top-right of the display during recording.

---

## Step 12 — Window setup (no toolbar, fullscreen support)

```python
WIN = "Thermal Camera - HIK HikCamera"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(WIN, THERMAL_W * SCALE, THERMAL_H * SCALE)
```

`WINDOW_GUI_NORMAL` removes the Qt toolbar (pan/zoom buttons). `WINDOW_NORMAL` allows
the window to be freely resized and toggled fullscreen.

Fullscreen toggle on **F**:
```python
fs = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN if fs != cv2.WINDOW_FULLSCREEN else cv2.WINDOW_NORMAL)
```

---

## File locations

| File | Purpose |
|------|---------|
| `/home/pi/thermal_viewer.py` | Main application |
| `/etc/modprobe.d/hikvision-thermal.conf` | UVC quirk (permanent, `options uvcvideo quirks=2`) |
| `~/.hikcam_cal.json` | Saved calibration (auto-created) |

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| Q | Quit |
| F | Toggle fullscreen |
| R | Start / stop video recording (raw + annotated) |
| S | Save screenshot as `thermal_YYYYMMDD_HHMMSS.png` |
| K | Set hot calibration point (99th-pct of frame = 80°C) |
| A | Set ambient calibration point (median of frame = 22°C) |
| 0 | Reset calibration |
| D | Diagnostic dump (raw u16 stats + cal params to terminal) |

---

## HUD layout

- **Top-left**: FPS, date, time
- **Top-right**: vendor/model, S/N, USB bus ID (+ pulsing REC indicator when recording)
- **Bottom-left** (reversed stack): build timestamp, frame count / hot%, CPU temp, cal status
- **Bottom-right**: scene min/mean/max °C
- **Centre**: green crosshair + centre-pixel temperature
- **Hot blob** (when locked): pulsing red bounding box, cross at centroid, yellow arrow from image centre, temperature label above box, `X:±N Y:±N [Ns]` below box

---

## Key constants to tune

| Constant | Default | Effect |
|----------|---------|--------|
| `SCALE` | 3 | Upscale factor (256×192 → 768×576) |
| `HOT_DISPLAY_PCT` | 90 | Colour the hottest `100-N`% of valid pixels |
| `BLOB_THRESHOLD_PCT` | 90 | Percentile threshold for blob candidate mask |
| `BLOB_MIN_AREA` | 9 | Minimum blob size in sensor pixels |
| `HotBlobTracker.CONFIRM_SECS` | 0.25 | Seconds of stability needed to lock |
| `HotBlobTracker.MAX_MISS_SECS` | 1.5 | Seconds to coast before dropping lock |
| `_cal_temp_hot` | 80.0 | Target °C for K calibration key |
| `_cal_temp_amb` | 22.0 | Target °C for A calibration key |

---

## Known issues / open questions

1. **Temperature formula accuracy**: Base formula `u16/100 − 273.15` confirmed empirically
   (BE/D=100 gives mean ≈ 26°C for room-temp scene). Absolute accuracy requires K/A calibration.
   The divisor may vary by firmware — the D key terminal output reveals the raw u16 range.

2. **UVC Extension Unit**: Camera has UVC XU (GUID `a29e7641-de04-47e3-8b2b-f4341aff003b`,
   unit ID 10, 15 controls). Control selector 4 returns `"2.0"`. May expose emissivity or
   factory calibration — not yet explored.

3. **Optical camera**: `/dev/video1` provides 640×360 optical. Can be added with a standard
   `cv2.VideoCapture` alongside the raw V4L2 path.
