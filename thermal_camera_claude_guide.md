# Hikvision USB Thermal Camera Viewer -- Complete Rebuild Guide

This document contains every detail needed to recreate `/home/pi/thermal_viewer.py` (1834 lines) from scratch. It covers hardware quirks, data formats, algorithms, neural network weights, and design rationale.

---

## 1. Project Overview

A real-time thermal camera viewer and target tracker for the Hikvision USB thermal camera module, running on a Raspberry Pi 5.

| Item | Value |
|------|-------|
| Device | HIK HikCamera |
| USB ID | `2bdf:0102` |
| Serial | F10615613 |
| Platform | Raspberry Pi 5, aarch64, Linux 6.12, Raspberry Pi OS Bookworm |
| Video node | `/dev/video0` (thermal), `/dev/video1` (optical, unused) |
| Sensor resolution | 256 x 192 pixels |
| Frame rate | ~25 fps |
| Language | Python 3, single file (`thermal_viewer.py`) |
| Dependencies | OpenCV (system `python3-opencv`), NumPy, ctypes, tkinter (for replay picker) |
| Build stamp | `BUILD_TIME = "2026-03-29 12:00"` |

The software captures raw thermal frames via V4L2, extracts per-pixel temperature data, displays a greyscale+colour thermal image with a neural-network-driven target tracker, and supports recording, replay, and temperature calibration.

---

## 2. Hardware Quirks and Setup

### 2.1 UVC Quirk (required or camera will not open)

The camera stalls when the UVC driver sends a `GET_MIN` probe control query (`0x82`), causing `EPIPE` and making all subsequent ioctls return `EIO`.

**Fix:** Set UVC quirk bit `0x02` (`UVC_QUIRK_PROBE_MINMAX`).

Runtime fix:
```bash
echo 2 | sudo tee /sys/module/uvcvideo/parameters/quirks
```

Permanent fix -- create `/etc/modprobe.d/hikvision-thermal.conf`:
```
options uvcvideo quirks=2
```

Reload: `sudo modprobe -r uvcvideo && sudo modprobe uvcvideo` (or reboot).

### 2.2 OpenCV System Package vs pip Wheel

The pip `opencv-python` wheel (manylinux) has a **non-functional V4L2 backend** on aarch64. The system package must be used:

```bash
pip3 uninstall opencv-python
sudo apt-get install python3-opencv   # 4.10.x on Bookworm
```

### 2.3 QT_OPENGL=software Requirement

Qt5 OpenCV on Pi uses OpenGL by default, which produces a blank window on the Pi 5 framebuffer. This environment variable **must** be set before `import cv2`:

```python
import os
os.environ.setdefault("QT_OPENGL", "software")
import cv2
```

Also avoid non-ASCII characters in window titles.

### 2.4 modprobe.d Config File

File: `/etc/modprobe.d/hikvision-thermal.conf`
```
options uvcvideo quirks=2
```

---

## 3. Camera Data Format -- The Key Discovery

### 3.1 Frame Layout: 256 x 392 YUYV

The camera advertises and delivers 256x392 YUYV frames. Each row is 512 bytes (256 pixels x 2 bytes/pixel in YUYV). OpenCV can only negotiate 256x192 (visual only), which is why raw V4L2 is required.

| Rows | Content |
|------|---------|
| 0 -- 191 | Visual thermal image (192 rows x 256 pixels, YUYV, black-hot) |
| 192 | Header row: image dimensions `(0x00, 0xC0=192, 0x01, 0x00=256)` |
| 193 | Magic sync row: `0xAA 0xBB 0xCC 0xDD` repeated (stored LE: `dd cc bb aa`) |
| 194 -- 195 | Unused / padding |
| **196 -- 387** | **Per-pixel temperature (192 rows x 256 values, big-endian uint16)** |
| 388 | Calibration footer |
| 389 -- 391 | Unused / padding |

### 3.2 Temperature Encoding

Each temperature row is 512 bytes. For pixel `(row, col)`:

```
u16 = (byte[2*col] << 8) | byte[2*col + 1]   # big-endian uint16
T(C) = u16 / 100.0 - 273.15
```

**The divisor is 100, NOT 64.** This was the key discovery. The Hikvision SDK documentation and most online references cite 64, but empirical testing proved 100 is correct for this camera:

- **With divisor 64:** mean temperature for a room-temp scene is ~176C (obviously wrong)
- **With divisor 100:** mean temperature is ~26C (correct for indoor ambient)

#### How This Was Determined Empirically

Testing was done by sweeping divisors from 40 to 200 and checking what fraction of pixels fell in the valid temperature range (-40C to +400C) and what the mean temperature was. Only divisor=100 with big-endian byte order produced a `valid_frac` of ~0.31 (matching the active sensor area) and a mean of ~26C.

The magic sync row (row 193) byte order is little-endian (`dd cc bb aa`), but the temperature uint16 values are big-endian. This asymmetry is consistent -- the camera packs raw uint16 values into the YUYV stream regardless of YUYV channel semantics.

### 3.3 Black-Hot Polarity and Inversion

The camera streams in **black-hot** mode: cold objects are bright, hot objects are dark in the raw Y channel. Must invert before display:

```python
visual = (255 - frame[:192, 0::2]).astype(np.uint8)
```

### 3.4 Border Pixel Trap (69% Dead Pixels)

Approximately 69% of the 256x192 grid consists of masked/dead border pixels with raw Y values of 14-26. After black-hot inversion, these become 229-241 -- the **brightest** values in the image. This is a critical trap:

**Never use the visual Y channel for blob/hot-region detection.** Always use the temperature array, where border pixels decode to T ~ -220C and are NaN-masked:

```python
T[(T < -40) | (T > 400)] = np.nan
```

---

## 4. Raw V4L2 Capture via ctypes

### 4.1 Why OpenCV Cannot Be Used

OpenCV's V4L2 backend will only negotiate 256x192 (the visual portion). To get the full 256x392 frame with temperature data, raw V4L2 ioctls through ctypes are required.

### 4.2 aarch64 Struct Sizes

The `v4l2_buffer` struct is **88 bytes** on aarch64 (not 68 as on 32-bit x86). This affects all buffer-related ioctl codes since the size is encoded in the ioctl number.

### 4.3 All ioctl Codes

```python
_libc = ctypes.CDLL("libc.so.6", use_errno=True)

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP            = 1
V4L2_PIX_FMT_YUYV           = 0x56595559

# Direction=3 (RW), type='V'=86=0x56, size=88 for buffer ioctls
VIDIOC_QUERYBUF  = (3<<30) | (ord('V')<<8) | 9  | (88<<16)
VIDIOC_QBUF      = (3<<30) | (ord('V')<<8) | 15 | (88<<16)
VIDIOC_DQBUF     = (3<<30) | (ord('V')<<8) | 17 | (88<<16)

# These are size-independent:
VIDIOC_S_FMT     = 0xC0D05605
VIDIOC_REQBUFS   = 0xC0145608
VIDIOC_STREAMON  = 0x40045612
VIDIOC_STREAMOFF = 0x40045613
```

### 4.4 v4l2_buffer Field Offsets (88-byte struct, aarch64)

```python
_BUF_OFF_INDEX  = 0    # buffer index (uint32)
_BUF_OFF_TYPE   = 4    # buffer type (uint32)
_BUF_OFF_USED   = 8    # bytesused (uint32)
_BUF_OFF_MEM    = 60   # memory type (uint32)
_BUF_OFF_MOFF   = 64   # m.offset for mmap (uint32)
_BUF_OFF_LEN    = 72   # length (uint32)
```

Helper to create a zeroed buffer struct:
```python
def _make_vbuf(idx=0):
    b = ctypes.create_string_buffer(88)
    struct.pack_into("II", b, _BUF_OFF_INDEX, idx, V4L2_BUF_TYPE_VIDEO_CAPTURE)
    struct.pack_into("I",  b, _BUF_OFF_MEM,   V4L2_MEMORY_MMAP)
    return b
```

### 4.5 mmap Buffer Cycle

1. `VIDIOC_S_FMT` -- request 256x392 YUYV format
2. `VIDIOC_REQBUFS` -- allocate N mmap buffers (default 3)
3. For each buffer: `VIDIOC_QUERYBUF` to get offset/length, `mmap()` the buffer, `VIDIOC_QBUF` to enqueue
4. `VIDIOC_STREAMON` to start capture
5. Main loop: `select()` (fd is O_NONBLOCK) -> `VIDIOC_DQBUF` -> copy data -> `VIDIOC_QBUF` to re-enqueue
6. `VIDIOC_STREAMOFF` on exit

`select()` is required because the fd is opened with `O_NONBLOCK`, so `DQBUF` returns `EAGAIN` without it.

### 4.6 ThermalCamera Class Design

```python
class ThermalCamera:
    FRAME_H        = 392
    TEMP_ROW_START = 196
    TEMP_ROW_END   = 388   # exclusive

    def __init__(self, device="/dev/video0", n_bufs=3):
        # Opens device, sets format, allocates mmap buffers, starts streaming

    def read(self, timeout=1.0):
        """Returns (visual_gray, temp_celsius, u16_raw) or (None, None, None)."""
        # visual_gray : uint8  (192, 256)  Y-channel, black-hot inverted
        # temp_celsius: float32(192, 256)  per-pixel C, NaN where invalid
        # u16_raw     : uint16 (192, 256)  raw encoded values (for calibration)

    def release(self):
        # STREAMOFF, close mmap buffers, close fd
```

Frame extraction in `read()`:
```python
frame = raw.reshape(392, 512)  # 256 YUYV pixels x 2 bytes = 512 bytes/row

# Visual: Y channel, inverted (black-hot -> white-hot)
visual = (255 - frame[:192, 0::2]).astype(np.uint8)

# Temperature: rows 196-387 as big-endian uint16
tr  = frame[196:388].reshape(192, 256, 2)
u16 = (tr[:,:,0].astype(np.uint16) << 8) | tr[:,:,1].astype(np.uint16)
T   = u16.astype(np.float32) / 100.0 - 273.15
T[(T < -40) | (T > 400)] = np.nan
```

---

## 5. Display System

### 5.1 Greyscale + Adaptive Colour

The display is greyscale everywhere except for genuinely hot pixels, which are shown in the Inferno colourmap. The colour threshold is **adaptive**: it uses whichever is more selective of:
- The `HOT_DISPLAY_PCT` percentile (90th) of valid temperature pixels
- Mean + 3 standard deviations of valid temperature pixels

```python
pct_thresh  = float(np.percentile(T_valid_disp, HOT_DISPLAY_PCT))
stat_thresh = float(T_valid_disp.mean() + 3.0 * max(T_valid_disp.std(), 0.1))
hot_thresh  = max(pct_thresh, stat_thresh)   # use whichever is higher (more selective)
```

This means in a uniform scene, very few pixels get colour. When there is a strong heat source, only it gets colour.

The combination is done at sensor resolution (256x192) before a single upscale:
```python
hot_mask = ((~np.isnan(T)) & (T >= hot_thresh)).astype(np.uint8)
combined = np.where(hot_mask[:,:,np.newaxis], colored, gray_bgr)
thermal  = cv2.resize(combined, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
```

Use `INTER_LINEAR` (not `INTER_NEAREST`) to avoid blocky appearance at 3x upscale.

### 5.2 Canvas Layout with Margins

The thermal image is placed inside a canvas with margins on all sides:

```python
SCALE     = 3          # 256x192 -> 768x576
MARGIN_TOP = 28        # pixels above thermal image for HUD text
MARGIN_BOT = 248       # pixels below for graphs + HUD text
MARGIN_L   = 8         # pixels left of thermal image
MARGIN_R   = 200       # right panel for candidate list
BG_COLOR   = (30, 30, 30)   # dark grey background
```

Canvas dimensions:
```python
canvas_w = MARGIN_L + THERMAL_W * SCALE + MARGIN_R   # 8 + 768 + 200 = 976
canvas_h = MARGIN_TOP + THERMAL_H * SCALE + MARGIN_BOT  # 28 + 576 + 248 = 852
ox, oy   = MARGIN_L, MARGIN_TOP   # thermal image origin in canvas
```

### 5.3 No Overlay on Thermal Image

All HUD text (FPS, datetime, device info, temperatures, build info) is drawn in the margins, **never** overlapping the thermal image itself. Only tracker elements (bounding boxes, crosshairs, arrows) are drawn on the thermal region.

### 5.4 Candidate Panel (Right Side)

The right margin contains a panel showing the top-N scored blobs with:
- Thumbnail crop from the thermal image
- Softmax probability bar (relative to top scorer, temperature=0.3)
- Feature summary line: area, motion, growth, has-moved
- Rejection reason for non-primary blobs (e.g. "too large", "static")
- Green border + "PRIMARY" label for the confirmed target
- Entries persist for `PANEL_STICKY` (3.0) seconds after disappearing, with fade

```python
PANEL_THUMB  = 40    # thumbnail size in pixels
PANEL_MAX    = 6     # max candidates shown
PANEL_STICKY = 3.0   # seconds of persistence
```

### 5.5 Position Graphs Below Image

Two time-series graphs are drawn below the thermal image:
- **X (Left/Right):** horizontal offset of locked target from image centre
- **Y (Up/Down):** vertical offset (inverted: positive = above centre)

```python
GRAPH_H    = 60    # height of each graph in pixels
GRAPH_GAP  = 6     # gap between graphs
GRAPH_SECS = 30    # seconds of history shown
```

Features:
- Symmetric y-axis (minimum +/-10), auto-scaling
- Centre line at zero
- Axis labels ("Left"/"Right", "Down"/"Up")
- Current value displayed at right edge in cyan
- Green triangle markers at timestamps when tracker acquired lock (target changes)
- Orange polyline trace

---

## 6. Neural Network Target Scorer (BlobScorer)

### 6.1 Architecture

A feedforward neural network: **7 inputs -> 10 hidden neurons (ReLU) -> 1 output (linear)**.

No training data was used. The weights were hand-crafted based on domain knowledge about what makes a good target (small, moving, growing, approaching).

### 6.2 All 7 Input Features

| Index | Name | Computation | Purpose |
|-------|------|-------------|---------|
| 0 | `size_inv` | `1.0 / (1.0 + area / 25.0)` | Smaller blobs score higher. area=25->0.5, area=100->0.2, area=1000->0.024 |
| 1 | `indep_motion` | `min(raw_flow * 5.0, 15.0) * size_inv * 5.0` | Size-gated independent motion (see 6.3) |
| 2 | `growth_rate` | Linear regression of area vs time for nearby blobs over recent history, normalised by mean area | Positive = blob is getting larger (approaching) |
| 3 | `temp_ratio` | `blob_t_mean / frame_max_T` | How hot relative to hottest thing in scene |
| 4 | `distinctness` | `(blob_t_mean - frame_mean_T) / frame_std_T` | Thermal contrast (z-score) |
| 5 | `centering` | `1.0 - dist_to_centre / max_dist` | How close to image centre (1=centre, 0=corner) |
| 6 | `has_moved` | `0.7 * prev_ema + 0.3 * indep_motion` | EMA of historical motion, keyed by spatial bucket `(cx//8, cy//8)`. Rewards targets that have been moving even if currently paused |

### 6.3 The Critical Insight: Size-Gated Motion

Raw Farneback optical flow values at 256x192 resolution are sub-pixel (typically 0.1--1.5 px/frame). This is too small for the neural network to differentiate meaningfully.

The solution is a two-stage amplification:
1. **5x raw amplification:** `raw_motion = min(indep_flow * 5.0, 15.0)` -- brings values into a usable range, capped at 15
2. **Size gating:** `indep_motion = raw_motion * size_inv * 5.0` -- multiplies by `size_inv` so that a small blob with any motion at all dominates over a large region with the same flow magnitude

This is essential because large warm background regions often show slight flow from camera vibration. Without size gating, these would outscore small genuine targets.

### 6.4 Weight Matrices with Neuron-by-Neuron Annotation

**W1 (10x7) -- Hidden layer weights:**

```python
W1 = np.array([
    #  sz    mot   grw   tmp   dst   ctr   mvd
    [ 3.0,  0.3,  0.2,  0.1,  1.0,  0.0,  0.5],  # H0: small + distinct
    [ 4.0, 10.0,  0.5,  0.0,  0.0,  0.0,  0.0],  # H1: small + motion (primary motion detector)
    [ 3.0,  6.0,  0.0,  0.0,  0.0,  0.0,  2.0],  # H2: small + motion + history
    [ 0.3,  0.5,  5.0,  0.0,  0.3,  0.0,  0.0],  # H3: growth-dominant
    [ 3.0,  4.0,  4.0,  0.0,  0.0,  2.0,  0.0],  # H4: small + motion + growth + centering
    [ 2.0,  3.0,  3.0,  0.0,  0.0,  4.0,  2.0],  # H5: approach pattern (move + center + grow)
    [ 4.0,  2.0,  2.0,  0.0,  0.5,  1.0,  1.0],  # H6: small + moving + growing
    [-1.0,  0.2,  0.1,  2.0,  1.5,  0.0,  0.0],  # H7: thermal fallback (penalises large size)
    [ 3.0,  5.0,  0.0,  0.0,  0.0,  0.0,  5.0],  # H8: small + has-moved (sticky memory)
    [ 0.5,  1.0,  0.5,  0.3,  0.5,  0.5,  0.3],  # H9: balanced (general-purpose)
], dtype=np.float32)
```

**b1 (10) -- Hidden biases:**
```python
b1 = np.array([0.0, -0.3, 0.0, 0.0, -0.5, -0.5, 0.0, 0.3, -0.3, 0.0], dtype=np.float32)
```

Negative biases on H1, H4, H5, H8 mean those neurons require a threshold of activation before firing -- they need strong input to activate, reducing false positives.

**W2 (1x10) -- Output weights:**
```python
W2 = np.array([[1.0, 5.0, 3.0, 2.5, 4.0, 5.0, 2.0, 0.4, 3.5, 0.8]], dtype=np.float32)
```

**b2 (1) -- Output bias:**
```python
b2 = np.array([0.0], dtype=np.float32)
```

### 6.5 What Each Hidden Neuron Specialises In

| Neuron | W2 weight | Specialisation |
|--------|-----------|----------------|
| H0 | 1.0 | Small + thermally distinct (static hot objects) |
| H1 | **5.0** | Primary motion detector: small blob + high motion. Highest-weighted motion path |
| H2 | 3.0 | Small + motion + motion history (sustains score for targets that have moved) |
| H3 | 2.5 | Growth-dominant (approaching target getting larger) |
| H4 | **4.0** | Approach pattern: small + moving + growing + centering on frame |
| H5 | **5.0** | Full approach signature: moving, centering, growing, with motion history |
| H6 | 2.0 | General small-moving-growing detector |
| H7 | 0.4 | Thermal fallback: fires for hot things even if large. Low output weight |
| H8 | 3.5 | Sticky motion memory: keeps score high for targets that moved recently |
| H9 | 0.8 | Balanced general-purpose feature. Low weight, acts as tiebreaker |

### 6.6 Feature Contribution Analysis via Perturbation

The `feature_contributions()` method measures each feature's marginal contribution by perturbing each input by +1.0 and measuring the output change:

```python
for i, name in enumerate(cls.FEATURE_NAMES):
    x2 = x.copy()
    x2[i] += 1.0
    h2 = np.maximum(0.0, cls.W1 @ x2 + cls.b1)
    contribs[name] = float((cls.W2 @ h2 + cls.b2)[0]) - base
```

This is displayed in the candidate panel to explain why one blob ranks above another.

---

## 7. Motion Analyser (MotionAnalyser)

### 7.1 Dense Optical Flow (Farneback)

Uses `cv2.calcOpticalFlowFarneback` at **sensor resolution** (256x192) on the grayscale visual frame:

```python
self._flow = cv2.calcOpticalFlowFarneback(
    self._prev_gray, gray, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
```

Performance: approximately 15ms per frame on Pi 5.

### 7.2 Background Subtraction via Median Flow

Background (camera) motion is estimated as the median flow across the entire frame:
```python
self.bg_vx = float(np.median(self._flow[:, :, 0]))
self.bg_vy = float(np.median(self._flow[:, :, 1]))
```

This robustly captures camera shake/pan because the background occupies the majority of the frame.

### 7.3 Per-Blob Independent Motion

For each blob, the mean flow within its bounding box is computed, then the background motion is subtracted:

```python
roi = self._flow[by:by+bh, bx:bx+bw]
bvx = float(np.mean(roi[:, :, 0])) - self.bg_vx
bvy = float(np.mean(roi[:, :, 1])) - self.bg_vy
b["indep_motion"] = (bvx**2 + bvy**2) ** 0.5
```

This gives independent motion magnitude in pixels/frame at sensor resolution.

---

## 8. Hot Blob Tracker (HotBlobTracker)

### 8.1 Two-State Design: SEARCHING -> LOCKED

**SEARCHING:** Score all blobs using the BlobScorer neural network. Track the top scorer. If the top-scoring blob remains stable (centroid drift within tolerance) for `CONFIRM_SECS` (0.5s), transition to LOCKED. Two additional gates must be passed before lock is acquired: the best blob must score at least `MIN_SCORE` (200.0) and must have independent motion of at least `MIN_MOTION` (0.8). These gates reject static hot regions that score well on thermal features alone.

**LOCKED:** Predict next position using EMA-smoothed velocity. Search within a radius around the prediction. Hold lock through occlusions. Only break lock if the target is truly gone or a dramatically better target appears.

### 8.2 SEARCHING State Details

- Maintains a history deque of `(time, cx, cy, score)` tuples
- Old entries (older than `CONFIRM_SECS + 0.5s`) are expired
- **MIN_SCORE gate:** If the best blob scores below `MIN_SCORE` (200.0), the history is cleared and no lock can be acquired. This eliminates false positives from blobs scoring in the 0-105 range (typical for static hot regions).
- **MIN_MOTION gate:** If the best blob's motion feature is below `MIN_MOTION` (0.8), the history is cleared. A static hot region is not a target, no matter how hot it is.
- **Drift tolerance:** normally `MAX_DRIFT_PX` (20 sensor px) per frame, but **relaxed to 2.5x (50px)** for blobs scoring above 500. This is critical because small, noisy targets can jitter in position between frames
- If drift exceeds the limit, history is cleared and confirmation timer restarts
- Lock is acquired when the history spans at least `CONFIRM_SECS` (0.5s)

### 8.3 LOCKED State Details

**Velocity prediction:**
```python
pred_x = self._lx + self._vx * dt
pred_y = self._ly + self._vy * dt
```

**EMA velocity smoothing** (alpha=0.75, higher = smoother, less jitter):
```python
self._vx = VEL_ALPHA * self._vx + (1 - VEL_ALPHA) * raw_vx
self._vy = VEL_ALPHA * self._vy + (1 - VEL_ALPHA) * raw_vy
```

**Search radius** expands with velocity uncertainty over time since last detection:
```python
radius = SEARCH_RADIUS + (abs(self._vx) + abs(self._vy)) * time_since
```

The base `SEARCH_RADIUS` is 20 sensor pixels (reduced from 35 to prevent jumping to nearby hot spots).

**Candidate scoring when locked** heavily penalises distance from the predicted position to prevent jumping to nearby hot spots:
```python
score = b["nn_score"] * 0.5 - dist * 1.5 + b["t_mean"] * 0.1
```

The distance penalty coefficient is 1.5 (increased from 0.2 in earlier versions). This means a blob 10px away from the prediction incurs a -15 penalty, requiring a substantially higher neural score to compete with a blob at the predicted position. Additionally, candidates with `nn_score < MIN_SCORE` (200.0) are rejected even in locked mode.

**Temperature gating:** candidate must have `t_mean >= frame_max_T * MIN_TEMP_RATIO` (0.50).

**Coasting through occlusion:** if no candidate is found, the predicted position advances using the established velocity but the lock is maintained for up to `MAX_MISS_SECS` (3.0s). The last confirmed blob continues to be displayed. If the coasting position exits the frame boundaries (5px margin), the lock releases immediately.

**Temperature EMA:** `self._lt = 0.8 * self._lt + 0.2 * best_blob["t_mean"]`

### 8.4 Trajectory-Based Lock Release

Two mechanisms release the lock based on target behaviour:

**Frame boundary exit:** When the predicted position moves outside the frame boundaries with a 5px margin, the lock releases immediately. This applies both to the initial prediction and to coasting positions:
```python
margin = 5
if (pred_x < -margin or pred_x > THERMAL_W + margin or
        pred_y < -margin or pred_y > THERMAL_H + margin):
    self._state = "SEARCHING"
```

**Stationarity detection:** After 2+ seconds of lock, if the target has become stationary (speed < 0.5 sensor px/s AND blob motion feature < 0.3), the lock releases. A real target should exhibit meaningful motion; a warm pixel stuck in place is not a target:
```python
speed = (self._vx**2 + self._vy**2) ** 0.5
blob_motion = best_blob.get("features", {}).get("motion", 0)
lock_age = now - self._lock_since
if lock_age > 2.0 and speed < 0.5 and blob_motion < 0.3:
    self._state = "SEARCHING"
```

### 8.5 Preemption

If the global best blob (highest `nn_score` across all blobs) scores `PREEMPT_RATIO` (5.0) times better than the currently locked target AND scores above 100, the lock is broken and the tracker returns to SEARCHING. The 5x ratio (increased from 3x) makes the lock very sticky -- only an overwhelmingly better target can break it. This prevents the tracker from getting distracted by transiently high-scoring blobs while maintaining a solid lock.

### 8.6 Sticky Blob List for Display Persistence

Blobs are tracked in a "sticky" list keyed by spatial bucket `(cx//6, cy//6)`. Entries persist for `PANEL_STICKY` (3.0) seconds after they are last seen. The list is sorted by score and capped at `PANEL_MAX` (6). This prevents the candidate panel from flickering.

### 8.7 Growth Rate Estimation

For each blob, the tracker maintains a size history: a deque of `(time, cx, cy, area)` entries (max 150). Growth rate is estimated via linear regression of area vs time for entries within 30 sensor pixels of the current blob, normalised by mean area:

```python
slope / max(mean_area, 1.0)   # fractional area change per second
```

Requires at least 3 data points spanning at least 0.1 seconds.

### 8.8 Key Constants

```python
CONFIRM_SECS    = 0.5     # seconds of stability to acquire lock (longer = fewer false locks)
MAX_DRIFT_PX    = 20      # max per-frame jump in SEARCHING (relaxed 2.5x for score>500)
SEARCH_RADIUS   = 20      # base search radius when LOCKED (sensor pixels) — tighter to prevent jumping
VEL_ALPHA       = 0.75    # velocity EMA smoothing (higher = smoother, less jitter)
MAX_MISS_SECS   = 3.0     # hold lock this long with no detection before dropping lock
MIN_TEMP_RATIO  = 0.50    # candidate must be >= 50% of frame max temperature
SIZE_HISTORY_N  = 150     # entries in size history deque
PREEMPT_RATIO   = 5.0     # break lock only if another blob scores 5x better
MIN_SCORE       = 200.0   # minimum nn_score to be considered a valid target
MIN_MOTION      = 0.8     # minimum motion feature to confirm lock (reject static blobs)
```

### 8.9 False Positive Elimination Summary

Static hot regions (e.g. warm walls, radiators, heat vents) typically produce `nn_score` values in the 0-105 range because they lack independent motion. The false positive elimination chain works as follows:

1. **MIN_SCORE (200.0):** Rejects any blob scoring below 200 from being considered a target, both in SEARCHING and LOCKED states. Static hot regions scoring 0-105 are immediately eliminated.
2. **MIN_MOTION (0.8):** Even if a static blob somehow scores above 200 (e.g. due to high thermal distinctness), the motion gate requires independent motion >= 0.8 before lock can be confirmed.
3. **Stationarity detection (locked):** If a locked target stops moving (speed < 0.5, motion < 0.3) for more than 2 seconds, the lock releases.
4. **PREEMPT_RATIO (5.0):** Lock only breaks for a blob scoring 5x better, preventing brief false detections from stealing lock.

---

## 9. Multi-Level Blob Finder

### 9.1 Why Simple Percentile Thresholding Fails

A single threshold (e.g. 90th percentile) merges small genuine targets with large warm background regions into a single connected component. The small target is lost inside the large blob, which then gets suppressed by `BLOB_MAX_AREA`.

### 9.2 Two-Level Approach

**Level 1 -- Percentile threshold** (`BLOB_THRESHOLD_PCT` = 90th percentile):
- Catches everything warm
- **With** morphological opening (3x3 rect kernel) to remove single-pixel noise
- Produces larger, merged blobs

**Level 2 -- Statistical outlier threshold** (mean + 2.5 * std):
- Isolates only the hottest peaks
- **Without** morphological opening -- this is critical because small genuine targets (3x3 to 5x5 pixels) would be destroyed by opening
- Produces smaller, more precise blobs

### 9.3 Selection Logic

```python
small_stat = [b for b in blobs_stat if BLOB_MIN_AREA < b["area"] <= BLOB_MAX_AREA]
small_pct  = [b for b in blobs_pct  if BLOB_MIN_AREA < b["area"] <= BLOB_MAX_AREA]

if small_stat:
    blobs = blobs_stat       # prefer statistical if it finds small targets
elif small_pct:
    blobs = blobs_pct        # fall back to percentile
```

### 9.4 Progressive Splitting for Oversized Blobs

If all detected blobs exceed `BLOB_MAX_AREA` (1500), the threshold is progressively raised through `[3.5, 4.5, 5.5]` sigma to try to split them:

```python
for extra_sigma in [3.5, 4.5, 5.5]:
    higher = frame_mean + extra_sigma * frame_std
    split = self._extract_blobs(T, valid, higher, open_morph=False)
    small_split = [b for b in split if b["area"] <= BLOB_MAX_AREA]
    if small_split:
        blobs = split
        break
```

### 9.5 BLOB_MAX_AREA Suppression

Blobs with area > `BLOB_MAX_AREA` (1500) are flagged `_suppressed = True` and given `nn_score = 0`. They remain in the blob list for display but cannot be selected as targets.

### 9.6 Blob Extraction

Uses `cv2.connectedComponentsWithStats` with 8-connectivity. Each blob stores:
- `cx`, `cy`: centroid (integer sensor coordinates)
- `t_mean`, `t_max`: mean and max temperature of valid pixels within the blob
- `bbox`: `(x, y, width, height)` bounding box
- `area`: pixel count

```python
BLOB_MIN_AREA = 9      # 3x3 minimum
BLOB_MAX_AREA = 1500   # suppress blobs larger than this
```

---

## 10. Temperature Calibration System

### 10.1 Baseline Formula

```python
T_raw = u16 / 100.0 - 273.15
```

### 10.2 Calibrated Formula

```python
T_cal = T_raw * _cal_slope + _cal_offset
```

Applied via `apply_cal()` to the entire temperature array each frame.

### 10.3 Key Press Behaviour

- **K key:** Captures the 99th percentile u16 value of the entire frame as the hot reference point. Target temperature: 80C (default). No aiming needed -- fill the frame with the hot object.
- **A key:** Captures the median u16 value of the entire frame as the ambient reference. Target temperature: 22C (default).
- **0 key:** Resets both references, slope=1, offset=0.

### 10.4 Single-Point Calibration (1-point, K or A only)

Treats the known point as a true Kelvin value and rescales the Kelvin axis:

```python
raw_K   = _cal_raw_hot / 100.0          # raw Kelvin value from camera
true_K  = _cal_temp_hot + 273.15        # true Kelvin
_cal_slope  = true_K / raw_K
_cal_offset = (_cal_slope - 1.0) * 273.15
```

This corrects both scale and zero-point simultaneously with a single reference, which is better than a simple offset correction because the error is multiplicative (wrong divisor), not additive.

### 10.5 Two-Point Calibration (K + A)

Full linear regression:
```python
t_hot_raw = _u16_to_raw_celsius(np.array(_cal_raw_hot, dtype=np.uint16)).item()
t_amb_raw = _u16_to_raw_celsius(np.array(_cal_raw_amb, dtype=np.uint16)).item()
_cal_slope  = (_cal_temp_hot - _cal_temp_amb) / (t_hot_raw - t_amb_raw)
_cal_offset = _cal_temp_hot - _cal_slope * t_hot_raw
```

Guard: if the raw temperatures of the two reference points differ by less than 0.5C, calibration is not applied (too noisy).

### 10.6 JSON Persistence

File: `~/.hikcam_cal.json`

Auto-saved after every K/A/0 press. Auto-loaded at startup.

```json
{
  "raw_hot": 35200,
  "raw_amb": 29500,
  "temp_hot": 80.0,
  "temp_amb": 22.0,
  "slope": 1.0234,
  "offset": -6.42
}
```

---

## 11. Recording System

### 11.1 Folder Structure

```
thermal_recordings/
  run_2026-03-28_14-30-00/
    raw_2026-03-28_14-30-00_build-2026-03-29_12-00.avi
    annotated_2026-03-28_14-30-00_build-2026-03-29_12-00.avi
```

### 11.2 Two Simultaneous Files

- **Raw file** (`raw_*.avi`): The thermal image **before** any HUD or blob overlay. Pure display image at `(disp_w, disp_h)` = `(768, 576)`.
- **Annotated file** (`annotated_*.avi`): Full canvas with all HUD, blob boxes, candidate panel, graphs at `(canvas_w, canvas_h)` = `(976, 852)`.

Both use MJPG codec at 25 fps.

### 11.3 Filename Format

`{type}_{timestamp}_build-{build_version}.avi`

Where build version has spaces replaced with underscores and colons replaced with hyphens.

### 11.4 Recording Control

Press R to start, R again to stop. A pulsing red dot and "REC Ns" counter are shown in the top-right corner during recording. Writers are also released on Q (quit).

---

## 12. Replay System

### 12.1 CLI Argument

```bash
python3 thermal_viewer.py --replay              # opens file picker
python3 thermal_viewer.py --replay path/to.avi   # replays specific file
```

Uses `argparse` with `nargs='?', const="__browse__"`.

### 12.2 GUI File Picker

A tkinter dialog (400x160, non-resizable) with four buttons arranged in a 2x2 grid:
- **Browse...**: opens a native file dialog (AVI/MP4 filter), starts in the latest recording directory
- **Last Recorded**: automatically selects the most recent raw recording
- **Last Replayed**: re-opens the last replayed file (disabled if none exists)
- **Cancel**: returns to live mode

### 12.3 O/P Keys During Live Mode

Pressing O or P during live mode opens the file picker. After replay closes, the live window is re-created and live capture resumes.

### 12.4 Synthesised Temperature for Non-Thermal Recordings

Since raw recordings do not contain temperature data, it is synthesised from grayscale:
```python
T = gray.astype(np.float32) * 0.5 + 10.0  # rough mapping: 10-137C
T[gray < 20] = np.nan  # mask very dark pixels (likely border)
```

This allows the tracker and blob finder to operate on replayed footage.

### 12.5 Replay Auto-Save

Replay mode automatically saves an annotated output video as `<source_basename>_REPLAY.avi` in the same directory as the source file. Uses MJPG codec at the source frame rate and the same canvas dimensions as the live display (976x852). Only non-paused frames are written.

```python
replay_out_path = os.path.join(src_dir, f"{src_base}_REPLAY.avi")
replay_writer   = cv2.VideoWriter(
    replay_out_path, cv2.VideoWriter_fourcc(*'MJPG'),
    fps_in, (canvas_w, canvas_h))
```

### 12.6 Slider / Trackbar for Scrubbing

A trackbar labelled "Frame" is created on the replay window for scrubbing through the video. A `slider_suppress` flag prevents callback loops when the slider position is updated programmatically (during normal playback):

```python
slider_suppress = [False]   # suppress callback during programmatic updates

def _on_slider(pos):
    if slider_suppress[0]:
        return
    slider_seeking[0] = True
    slider_target[0]  = pos
```

When the user drags the slider, the video seeks to that frame and the tracker is reset (new `HotBlobTracker()` instance). Playback pauses automatically on seek.

### 12.7 Last-Replayed File Persistence

The last-replayed file path is persisted to `~/.hikcam_last_replay`. This is loaded at module import time and used to populate the "Last Replayed" button in the file picker.

```python
_LAST_REPLAY_FILE = os.path.expanduser("~/.hikcam_last_replay")
```

### 12.8 O Key in Replay Mode

Pressing O during replay opens the file picker. If a new file is selected, the current replay writer is released and saved, the current video is closed, and a new `replay_main()` call is made recursively.

### 12.9 Pause/Step Controls

- **Space**: toggle pause/play
- **. or >**: step forward one frame (auto-pauses)
- **, or <**: step backward one frame (seeks in video, auto-pauses)
- **Q**: quit replay
- **F**: toggle fullscreen

---

## 13. Coordinate Convention

Image coordinates have Y increasing downward. The displayed target offsets use:
- **X positive = right** of image centre
- **Y positive = above** image centre (inverted from image coords)

```python
dx =   blob_cx - THERMAL_W // 2
dy = -(blob_cy - THERMAL_H // 2)
# Displayed as: X:+14 Y:-7
```

---

## 14. Key Lessons Learned / Pitfalls

1. **Temperature divisor is 100, not 64.** The standard Hikvision SDK documentation and many online references cite 64 as the divisor. For this camera, 64 gives mean temperatures of ~176C for a room-temperature scene. The correct divisor is 100 (verified empirically by sweeping 40-200 and checking valid pixel fractions and mean temperatures).

2. **Border pixels appear as hottest after black-hot inversion.** ~69% of the sensor grid is dead/masked border pixels with Y values of 14-26. After inversion (255 - Y), these become 229-241 -- brighter than any real thermal feature. Never use the visual Y channel for blob detection. Always use the temperature array where these pixels decode to -220C and are NaN-masked.

3. **Raw Farneback flow at 256x192 is sub-pixel.** At this low resolution, optical flow values are typically 0.1-1.5 px/frame. The neural network needs values amplified 5x AND gated by blob size (multiplied by `size_inv * 5.0`) so that small moving blobs produce scores 10-50x higher than large background regions with similar raw flow.

4. **Morphological opening destroys tiny targets.** A 3x3 opening kernel removes blobs smaller than ~3x3 pixels. Since genuine targets at distance may be exactly this size, morphological opening must be **skipped** at the statistical threshold level (mean+2.5sigma). Only apply opening at the lower percentile threshold.

5. **Lock preemption is essential but must be conservative.** The 5x preemption check prevents the tracker from getting stuck on large static warm blobs while ensuring the lock doesn't break for transiently high-scoring blobs. The ratio was increased from 3x to 5x to make the lock stickier after testing showed false lock-breaks.

6. **WINDOW_GUI_NORMAL removes the Qt toolbar.** Without this flag, OpenCV's Qt backend adds a pan/zoom/save toolbar that wastes screen space and can interfere with keyboard shortcuts.

7. **Drift tolerance must be relaxed for high-scoring noisy small targets.** Small targets (3-5px) at 256x192 resolution have noisy centroid positions. At normal drift tolerance (20px), the confirmation timer keeps resetting. For blobs scoring above 500, drift tolerance is increased to 50px (2.5x).

8. **v4l2_buffer struct is 88 bytes on aarch64, not 68.** This affects QUERYBUF, QBUF, and DQBUF ioctl codes since the struct size is encoded in the ioctl number. Getting this wrong produces silent failures or `EINVAL`.

9. **select() is required before DQBUF.** The fd is opened O_NONBLOCK, so DQBUF returns EAGAIN without it.

10. **QT_OPENGL must be set before import cv2.** Setting it afterward has no effect.

11. **MIN_SCORE and MIN_MOTION are the primary false positive defence.** Without these gates, the tracker would lock onto static hot regions (scoring 0-105) that happen to persist for CONFIRM_SECS. The MIN_SCORE threshold of 200 and MIN_MOTION threshold of 0.8 ensure only genuinely moving targets with strong neural scores can acquire lock.

12. **Stickier lock prevents target-hopping.** The distance penalty in locked scoring (1.5 per sensor pixel) combined with a tight search radius (20px) prevents the tracker from jumping between nearby hot spots. Previously with a radius of 35 and penalty of 0.2, the tracker would hop between adjacent blobs.

---

## 15. All Constants and Their Tuning Rationale

### Camera/Hardware

| Constant | Value | Rationale |
|----------|-------|-----------|
| `DEVICE` | `/dev/video0` | Thermal camera video node |
| `THERMAL_W` | 256 | Sensor width |
| `THERMAL_H` | 192 | Sensor height |
| `FRAME_H` | 392 | Full frame height (192 visual + 4 header + 192 temp + 4 footer) |
| `TEMP_ROW_START` | 196 | First temperature data row |
| `TEMP_ROW_END` | 388 | First row after temperature data (exclusive) |

### Display

| Constant | Value | Rationale |
|----------|-------|-----------|
| `SCALE` | 3 | 256x192 -> 768x576, good balance of detail and performance |
| `MARGIN_TOP` | 28 | One line of HUD text above image |
| `MARGIN_BOT` | 248 | Two graphs (60px each + 6px gap) + 4 rows of text below |
| `MARGIN_L` | 8 | Small left padding |
| `MARGIN_R` | 200 | Wide enough for candidate panel thumbnails + text |
| `BG_COLOR` | (30,30,30) | Dark grey, not black -- provides contrast without being harsh |
| `PANEL_THUMB` | 40 | Thumbnail size -- large enough to see blob shape |
| `PANEL_MAX` | 6 | Max candidates -- more would overcrowd the panel |
| `PANEL_STICKY` | 3.0 | Seconds of display persistence -- prevents flickering |
| `GRAPH_H` | 60 | Graph height -- tall enough to read, short enough to fit |
| `GRAPH_GAP` | 6 | Small gap between X and Y graphs |
| `GRAPH_SECS` | 30 | History window -- long enough to see approach patterns |
| `HOT_DISPLAY_PCT` | 90 | Top 10% of pixels shown in colour |

### Blob Detection

| Constant | Value | Rationale |
|----------|-------|-----------|
| `BLOB_MIN_AREA` | 9 | 3x3 minimum -- smaller is noise |
| `BLOB_MAX_AREA` | 1500 | Blobs above this are background/structures, not targets |
| `BLOB_THRESHOLD_PCT` | 90 | Percentile for Level 1 blob finding |
| Statistical threshold | mean + 2.5 sigma | Level 2 -- isolates only genuine outliers |
| Progressive split sigmas | 3.5, 4.5, 5.5 | Increasing aggressiveness to split merged blobs |

### Tracker

| Constant | Value | Rationale |
|----------|-------|-----------|
| `CONFIRM_SECS` | 0.5 | Pre-lock stability window -- long enough to filter transients and false positives |
| `MAX_DRIFT_PX` | 20 | Normal drift limit -- prevents jumping between blobs |
| High-score drift multiplier | 2.5x | Relaxed to 50px for blobs scoring >500 (noisy small targets) |
| `SEARCH_RADIUS` | 20 | Tight base radius -- prevents jumping to nearby hot spots |
| `VEL_ALPHA` | 0.75 | Smooth velocity -- higher means more inertia, less jitter |
| `MAX_MISS_SECS` | 3.0 | Coast through occlusions up to 3 seconds |
| `MIN_TEMP_RATIO` | 0.50 | Candidate must be at least half as hot as the hottest pixel |
| `SIZE_HISTORY_N` | 150 | ~6 seconds at 25fps for growth rate estimation |
| `PREEMPT_RATIO` | 5.0 | Break lock only if another blob scores 5x better -- very sticky |
| `MIN_SCORE` | 200.0 | Minimum nn_score to consider a blob as a valid target |
| `MIN_MOTION` | 0.8 | Minimum motion feature before confirming lock |
| Locked scoring formula | `score = nn_score * 0.5 - dist * 1.5 + t_mean * 0.1` | Heavily penalises distance from predicted position |
| Frame exit margin | 5 px | Boundary for trajectory-based lock release |
| Stationarity: speed threshold | 0.5 sensor px/s | Below this, target is considered stationary |
| Stationarity: motion threshold | 0.3 | Below this, blob has no meaningful optical flow |
| Stationarity: min lock age | 2.0 seconds | Don't release lock for stationarity until 2s elapsed |

### Motion

| Constant | Value | Rationale |
|----------|-------|-----------|
| Farneback pyr_scale | 0.5 | Standard pyramid scaling |
| Farneback levels | 3 | Enough for the resolution |
| Farneback winsize | 15 | Large window compensates for low resolution |
| Farneback iterations | 2 | Balance speed vs accuracy |
| Farneback poly_n | 5 | Standard polynomial expansion |
| Farneback poly_sigma | 1.1 | Standard Gaussian sigma |
| Raw motion amplifier | 5.0 | Brings sub-pixel flow into usable range |
| Motion cap | 15.0 | Prevents extreme values from dominating |
| Size gate multiplier | 5.0 | Makes small blobs' motion 5x more impactful |
| Motion EMA alpha | 0.7 old / 0.3 new | Decays slowly so motion history persists |
| Motion EMA bucket size | 8 | Spatial grouping for has-moved tracking |
| Motion EMA cleanup threshold | >200 entries, prune <0.01 | Prevents unbounded memory growth |

### Calibration

| Constant | Value | Rationale |
|----------|-------|-----------|
| `_cal_temp_hot` | 80.0 | Typical boiling kettle surface temperature |
| `_cal_temp_amb` | 22.0 | Typical indoor ambient temperature |
| K key method | 99th percentile | Robust against single hot pixels, captures the hot object |
| A key method | median | Robust against outliers, captures the dominant scene temperature |
| Min raw difference guard | 0.5C | Prevents division by near-zero in 2-point calibration |

---

## 16. Full Keyboard Reference

| Key | Action |
|-----|--------|
| Q | Quit application |
| F | Toggle fullscreen |
| T | Toggle temperature display (centre crosshair temp, scene min/mean/max) |
| R | Start/stop video recording (raw + annotated) |
| S | Save screenshot as `thermal_YYYYMMDD_HHMMSS.png` |
| O / P | Open replay file picker (during live mode) |
| K | Set hot calibration point (99th percentile u16 = 80C) |
| A | Set ambient calibration point (median u16 = 22C) |
| 0 | Reset calibration (slope=1, offset=0) |
| D | Diagnostic dump to terminal (raw u16 stats, calibration params) |

### Replay-only keys

| Key | Action |
|-----|--------|
| Space | Pause/play toggle |
| . or > | Step forward one frame |
| , or < | Step backward one frame |
| O | Open a different recording file for replay |
| Q | Exit replay (return to live if launched from O/P key) |
| F | Toggle fullscreen |
| Slider | Scrub to any frame position |

---

## 17. File Locations

| File | Purpose |
|------|---------|
| `/home/pi/thermal_viewer.py` | Main application (1834 lines) |
| `/etc/modprobe.d/hikvision-thermal.conf` | UVC quirk (`options uvcvideo quirks=2`) |
| `~/.hikcam_cal.json` | Saved calibration (auto-created on K/A/0) |
| `~/.hikcam_last_replay` | Last-replayed file path (for "Last Replayed" button) |
| `thermal_recordings/run_YYYY-MM-DD_HH-MM-SS/` | Recording output folders (created in cwd) |

### Dependencies

```bash
sudo apt-get install python3-opencv   # system OpenCV with working V4L2 backend
# numpy comes with python3-opencv
# tkinter: sudo apt-get install python3-tk  (for replay file picker)
```

### UVC Extension Unit (unexplored)

The camera has a UVC Extension Unit: GUID `a29e7641-de04-47e3-8b2b-f4341aff003b`, unit ID 10, 15 controls. Control selector 4 returns `"2.0"`. May expose emissivity or factory calibration settings.

---

## Appendix: Complete Import List

```python
import os
os.environ.setdefault("QT_OPENGL", "software")

import cv2
import ctypes
import fcntl
import json
import mmap
import numpy as np
import select
import struct
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
```

For the replay file picker:
```python
import tkinter as tk
from tkinter import filedialog, ttk
```

And for CLI:
```python
import argparse
```

---

## Appendix: HUD Layout Detail

**Top margin (MARGIN_TOP = 28px):**
- Left: `FPS: 25.0   2026-03-28  14:30:00.123`
- Right: `HIK HikCamera  S/N:F10615613  Bus 001 Dev 003`
- Far right (when recording): pulsing red dot + `REC 15s`

**Bottom margin (MARGIN_BOT = 248px):**
- Position graphs occupy the first ~132px (two 60px graphs + 6px gap + 6px top padding)
- Row 1 (below graphs): scene temperature `Scene: 18.2 / 22.1 / 45.3 C` (left), calibration status (right)
- Row 2: `Thermal 256x192 @ 25fps   Hot top 10%   Frame: 1234` (left), `CPU 52.3C` (right)
- Row 3: `Build: 2026-03-29 12:00` (grey, smaller)
- Row 4: `px min:23  max:198` (grey, smaller)

**On the thermal image:**
- Green crosshair at image centre + temperature label (when T toggled on)
- Red pulsing bounding box around locked target + cross at centroid
- Yellow arrow from image centre to target centroid
- Temperature label above target box (when T toggled on)
- `X:+14 Y:-7  [2.3s]` below target box
- Blue dotted rectangles around non-primary candidate blobs
- "locking..." progress bar during SEARCHING confirmation window

**Right panel (MARGIN_R = 200px):**
- "Candidates" header
- Up to 6 entries, each with: thumbnail, probability bar, feature stats, rejection reason or "PRIMARY"

---

## Appendix: Window Setup

```python
WIN = "Thermal Camera - HIK HikCamera"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(WIN, canvas_w, canvas_h)
```

`WINDOW_GUI_NORMAL` (flag 0x10) removes the Qt toolbar. `WINDOW_NORMAL` allows free resizing and fullscreen toggle.

Fullscreen toggle:
```python
fs = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN if fs != cv2.WINDOW_FULLSCREEN else cv2.WINDOW_NORMAL)
```

---

## Appendix: Drawing Helpers

### Text with shadow
```python
def draw_text(img, text, pos, scale=0.55, thickness=1, color=(255,255,255), shadow=True):
    x, y = pos
    if shadow:
        cv2.putText(img, text, (x+1, y+1), FONT, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
```

### Dotted rectangle
The `draw_dotted_rect` function draws four edges with alternating on/off segments (gap=6 pixels). Used for secondary (non-primary) blob candidates.

### Colour bar
```python
def build_colorbar(height, colormap, width=22):
    bar = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    return cv2.applyColorMap(np.repeat(bar, width, axis=1), colormap)
```

### CPU temperature
Read from `/sys/class/thermal/thermal_zone0/temp`, divided by 1000 to get Celsius.

### USB bus info
Parsed from `lsusb` output, searching for `2bdf:0102` or `HikCamera`.

---

## Appendix: Rejection Reason Logic

When a blob is not the primary target, the candidate panel shows a human-readable reason. This is computed by comparing its features against the best blob's features:

| Condition | Reason shown |
|-----------|-------------|
| `size < 0.3` and best `size > 0.5` | "too large" |
| `motion < 0.3` and best `motion > 0.5` | "static" |
| `moved < 0.1` and best `moved > 0.3` | "no motion history" |
| `growth < 0.05` and best `growth > 0.1` | "not growing" |
| `center < 0.3` and best `center > 0.6` | "off-centre" |
| `distinct < best_distinct * 0.5` | "low contrast" |
| None of the above | Uses perturbation analysis to find the worst feature |

Maximum 2 reasons are shown, joined by "; ".

---

## Appendix: Softmax Probability Bars

The candidate panel shows relative probabilities using a softmax with temperature 0.3 (inverted, so `s * 0.3`):

```python
scores = [b["nn_score"] for b in all_blobs]
shifted = [s - max(scores) for s in scores]        # subtract max for numerical stability
exps = [np.exp(s * 0.3) for s in shifted]           # temperature = 1/0.3 ~ 3.3
exp_sum = sum(exps)
probs = [e / exp_sum for e in exps]
```

The 0.3 multiplier makes the distribution sharper -- the top scorer gets a dominant probability while lower scorers get small probabilities. This is displayed as a filled bar and percentage.
