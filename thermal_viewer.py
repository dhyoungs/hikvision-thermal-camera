#!/usr/bin/env python3
"""
Hikvision USB Thermal Camera Viewer
Device: HikCamera (HIK, 2bdf:0102)
Captures 256×392 YUYV frames:
  rows   0-191 : 8-bit visual thermal image
  rows 196-387 : per-pixel absolute temperature as big-endian uint16
                 formula: T(°C) = value / 64.0 - 273.15
"""

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

# ── Build info ────────────────────────────────────────────────────────────────
BUILD_TIME = "2026-03-29 12:00"

# ── Camera config ──────────────────────────────────────────────────────────────
DEVICE        = "/dev/video0"
THERMAL_W     = 256
THERMAL_H     = 192
SERIAL        = "F10615613"
VENDOR        = "HIK"
MODEL         = "HikCamera"

# ── Display config ─────────────────────────────────────────────────────────────
FONT          = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_COLOR = (255, 255, 255)
SHADOW_COLOR  = (0, 0, 0)
SCALE         = 3
MARGIN_TOP    = 28       # pixels above thermal image for HUD
MARGIN_BOT    = 248      # pixels below thermal image for graphs + HUD text
MARGIN_L      = 8        # pixels left of thermal image
MARGIN_R      = 200      # right panel for candidate list
BG_COLOR      = (30, 30, 30)   # dark grey background for margins
PANEL_THUMB   = 40       # thumbnail size in candidate panel
PANEL_MAX     = 6        # max candidates shown
PANEL_STICKY  = 3.0      # seconds a candidate stays in the panel after disappearing
GRAPH_H       = 60       # height of each position graph in pixels
GRAPH_GAP     = 6        # gap between graphs
GRAPH_SECS    = 30       # seconds of history shown

# ── Hot blob / display config ──────────────────────────────────────────────────
BLOB_MIN_AREA      = 9       # minimum blob area in sensor pixels (3×3)
BLOB_MAX_AREA      = 1500    # blobs larger than this are suppressed (not a target)
BLOB_THRESHOLD_PCT = 90      # percentile of valid-T pixels for blob candidate mask
HOT_DISPLAY_PCT    = 90      # top N% of valid pixels shown in colour; rest greyscale

# ── Temperature calibration (2-point linear) ───────────────────────────────────
# Press 'K' (crosshair on kettle ~100°C) and 'A' (crosshair on ambient ~22°C)
# to calibrate.  Press '0' to reset.  Calibration is auto-saved/loaded from file.
_cal_raw_hot  = None
_cal_raw_amb  = None
_cal_temp_hot = 80.0
_cal_temp_amb = 22.0
_cal_slope    = 1.0
_cal_offset   = 0.0
CAL_FILE      = os.path.expanduser("~/.hikcam_cal.json")


def _u16_to_raw_celsius(u16):
    """Baseline formula (before user calibration)."""
    return u16.astype(np.float32) / 100.0 - 273.15


def _update_calibration():
    """
    Recomputes _cal_slope / _cal_offset from whatever reference points are set.

    Single-point strategy: the camera formula T = u16/64 - 273.15 may have a
    wrong divisor (scale), not just a wrong zero-point.  Fixing only the offset
    (slope=1) would make one temperature right but everything else wrong.
    Instead we treat the single known point as a true Kelvin value and rescale
    so that the raw Kelvin (u16/64) maps to the true Kelvin at that point.
    This corrects both scale and offset simultaneously with one reference.

    Two-point: full linear regression, correct for any affine error.
    """
    global _cal_slope, _cal_offset
    if _cal_raw_hot is None and _cal_raw_amb is None:
        _cal_slope, _cal_offset = 1.0, 0.0
        return

    if _cal_raw_hot is not None and _cal_raw_amb is None:
        # 1-point using kettle reference
        # raw_K = u16/64  (raw Kelvin)
        # true_K = _cal_temp_hot + 273.15
        # We want: slope * raw_K + offset_K = true_K  with offset_K = 0
        # i.e. slope = true_K / raw_K  (rescale Kelvin axis)
        raw_K  = _cal_raw_hot / 100.0
        true_K = _cal_temp_hot + 273.15
        _cal_slope  = true_K / raw_K
        _cal_offset = -273.15 * _cal_slope + 273.15  # keep °C = K - 273.15 correct
        # Simplify: T_cal = (u16/64) * slope - 273.15
        # Apply as post-formula correction: T_base = u16/64 - 273.15
        # T_cal = T_base * slope + slope*273.15 - 273.15  ← encoded below
        _cal_offset = (_cal_slope - 1.0) * 273.15
        print(f"  1-pt cal (kettle={_cal_temp_hot:.0f}°C): slope={_cal_slope:.4f}  offset={_cal_offset:+.2f}°C")
        return

    if _cal_raw_hot is None and _cal_raw_amb is not None:
        raw_K  = _cal_raw_amb / 100.0
        true_K = _cal_temp_amb + 273.15
        _cal_slope  = true_K / raw_K
        _cal_offset = (_cal_slope - 1.0) * 273.15
        print(f"  1-pt cal (ambient={_cal_temp_amb:.0f}°C): slope={_cal_slope:.4f}  offset={_cal_offset:+.2f}°C")
        return

    # 2-point: full linear calibration
    t_hot_raw = _u16_to_raw_celsius(np.array(_cal_raw_hot, dtype=np.uint16)).item()
    t_amb_raw = _u16_to_raw_celsius(np.array(_cal_raw_amb, dtype=np.uint16)).item()
    if abs(t_hot_raw - t_amb_raw) < 0.5:
        print("  WARNING: hot and ambient raw temperatures too similar — calibration not applied")
        _cal_slope, _cal_offset = 1.0, 0.0
        return
    _cal_slope  = (_cal_temp_hot - _cal_temp_amb) / (t_hot_raw - t_amb_raw)
    _cal_offset = _cal_temp_hot - _cal_slope * t_hot_raw
    print(f"  2-pt cal: slope={_cal_slope:.4f}  offset={_cal_offset:+.2f}°C")
    print(f"    kettle check: raw {t_hot_raw:.1f}°C → {apply_cal_scalar(t_hot_raw):.1f}°C  "
          f"ambient check: raw {t_amb_raw:.1f}°C → {apply_cal_scalar(t_amb_raw):.1f}°C")


def apply_cal_scalar(t):
    return t * _cal_slope + _cal_offset


def save_calibration():
    data = {
        "raw_hot": _cal_raw_hot, "raw_amb": _cal_raw_amb,
        "temp_hot": _cal_temp_hot, "temp_amb": _cal_temp_amb,
        "slope": _cal_slope, "offset": _cal_offset,
    }
    try:
        with open(CAL_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Calibration saved → {CAL_FILE}")
    except Exception as e:
        print(f"  WARNING: could not save calibration: {e}")


def load_calibration():
    global _cal_raw_hot, _cal_raw_amb, _cal_temp_hot, _cal_temp_amb, _cal_slope, _cal_offset
    if not os.path.exists(CAL_FILE):
        return
    try:
        with open(CAL_FILE) as f:
            d = json.load(f)
        _cal_raw_hot  = d.get("raw_hot")
        _cal_raw_amb  = d.get("raw_amb")
        _cal_temp_hot = d.get("temp_hot", 80.0)
        _cal_temp_amb = d.get("temp_amb", 22.0)
        _cal_slope    = d.get("slope",  1.0)
        _cal_offset   = d.get("offset", 0.0)
        print(f"  Loaded calibration from {CAL_FILE}: "
              f"slope={_cal_slope:.4f}  offset={_cal_offset:+.2f}°C")
    except Exception as e:
        print(f"  WARNING: could not load calibration: {e}")


def apply_cal(T):
    """Apply current calibration to a temperature array."""
    return T * _cal_slope + _cal_offset


def cal_status_str():
    if _cal_raw_hot is None and _cal_raw_amb is None:
        return "cal:none"
    if _cal_raw_hot is not None and _cal_raw_amb is None:
        return f"cal:1pt(K={_cal_temp_hot:.0f}C)"
    if _cal_raw_hot is None and _cal_raw_amb is not None:
        return f"cal:1pt(A={_cal_temp_amb:.0f}C)"
    return f"cal:2pt s={_cal_slope:.3f} b={_cal_offset:+.1f}C"

# ── V4L2 raw capture constants (aarch64 64-bit layout) ────────────────────────
_libc = ctypes.CDLL("libc.so.6", use_errno=True)

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP            = 1
V4L2_PIX_FMT_YUYV           = 0x56595559

VIDIOC_S_FMT    = 0xC0D05605
VIDIOC_REQBUFS  = 0xC0145608
VIDIOC_QUERYBUF = (3<<30)|(ord('V')<<8)|9 |(88<<16)
VIDIOC_QBUF     = (3<<30)|(ord('V')<<8)|15|(88<<16)
VIDIOC_DQBUF    = (3<<30)|(ord('V')<<8)|17|(88<<16)
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF= 0x40045613

# v4l2_buffer field offsets for 88-byte struct on aarch64
_BUF_OFF_INDEX  = 0
_BUF_OFF_TYPE   = 4
_BUF_OFF_USED   = 8
_BUF_OFF_MEM    = 60
_BUF_OFF_MOFF   = 64
_BUF_OFF_LEN    = 72


def _make_vbuf(idx=0):
    b = ctypes.create_string_buffer(88)
    struct.pack_into("II", b, _BUF_OFF_INDEX, idx, V4L2_BUF_TYPE_VIDEO_CAPTURE)
    struct.pack_into("I",  b, _BUF_OFF_MEM,   V4L2_MEMORY_MMAP)
    return b


# ── Raw V4L2 camera class ──────────────────────────────────────────────────────
class ThermalCamera:
    """Opens /dev/videoN in raw V4L2 mode at 256×392 YUYV for temperature data."""

    FRAME_H  = 392
    TEMP_ROW_START = 196
    TEMP_ROW_END   = 388   # exclusive

    def __init__(self, device=DEVICE, n_bufs=3):
        self._fd    = None
        self._bufs  = []
        self._stype = struct.pack("I", V4L2_BUF_TYPE_VIDEO_CAPTURE)
        self._open(device, n_bufs)

    def _open(self, device, n_bufs):
        self._fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)

        # Set format: 256×392 YUYV
        fmt = bytearray(208)
        struct.pack_into("III",  fmt, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE, 0, 0)
        struct.pack_into("IIII", fmt, 8, THERMAL_W, self.FRAME_H, 1, V4L2_PIX_FMT_YUYV)
        fcntl.ioctl(self._fd, VIDIOC_S_FMT, fmt)

        # Request mmap buffers
        reqbuf = bytearray(20)
        struct.pack_into("III", reqbuf, 0, n_bufs, V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP)
        fcntl.ioctl(self._fd, VIDIOC_REQBUFS, reqbuf)

        # Query, mmap, and queue each buffer
        for i in range(n_bufs):
            qb = _make_vbuf(i)
            _libc.ioctl(self._fd, VIDIOC_QUERYBUF, qb)
            length = struct.unpack_from("I", qb, _BUF_OFF_LEN)[0]
            offset = struct.unpack_from("I", qb, _BUF_OFF_MOFF)[0]
            buf = mmap.mmap(self._fd, length, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=offset)
            self._bufs.append((buf, length))
            _libc.ioctl(self._fd, VIDIOC_QBUF, _make_vbuf(i))

        # Start streaming
        fcntl.ioctl(self._fd, VIDIOC_STREAMON, bytearray(self._stype))

    def read(self, timeout=1.0):
        """
        Returns (visual_gray, temp_celsius, u16_raw) or (None, None, None) on timeout.
        visual_gray : uint8  (192, 256)  Y-channel of the thermal image
        temp_celsius: float32(192, 256)  per-pixel °C (NaN where invalid)
        u16_raw     : uint16 (192, 256)  raw encoded temperature values
        """
        r, _, _ = select.select([self._fd], [], [], timeout)
        if not r:
            return None, None, None

        dq = _make_vbuf(0)
        _libc.ioctl(self._fd, VIDIOC_DQBUF, dq)
        idx  = struct.unpack_from("I", dq, _BUF_OFF_INDEX)[0]
        used = struct.unpack_from("I", dq, _BUF_OFF_USED)[0]

        buf, _ = self._bufs[idx]
        buf.seek(0)
        raw   = np.frombuffer(buf.read(used), dtype=np.uint8).copy()

        # Re-queue immediately
        _libc.ioctl(self._fd, VIDIOC_QBUF, _make_vbuf(idx))

        frame = raw.reshape(self.FRAME_H, THERMAL_W * 2)

        # Visual: Y channel of rows 0-191.
        # Camera streams in black-hot polarity (cold=bright, hot=dark), so invert
        # so that hot=bright throughout — correct for Inferno colormap and blob detection.
        visual = (255 - frame[:THERMAL_H, 0::2]).astype(np.uint8)

        # Temperature: rows 196-387 as big-endian uint16
        tr  = frame[self.TEMP_ROW_START:self.TEMP_ROW_END].reshape(THERMAL_H, THERMAL_W, 2)
        u16 = (tr[:, :, 0].astype(np.uint16) << 8 | tr[:, :, 1].astype(np.uint16))
        T   = u16.astype(np.float32) / 100.0 - 273.15

        # Mark out-of-range pixels as NaN
        T[(T < -40) | (T > 400)] = np.nan

        return visual, T, u16

    def release(self):
        if self._fd is not None:
            try:
                fcntl.ioctl(self._fd, VIDIOC_STREAMOFF, bytearray(self._stype))
            except Exception:
                pass
            for buf, _ in self._bufs:
                buf.close()
            os.close(self._fd)
            self._fd = None


# ── Motion analyser (dense optical flow) ──────────────────────────────────────
class MotionAnalyser:
    """Computes inter-frame optical flow and annotates blobs with motion features."""

    def __init__(self):
        self._prev_gray = None
        self._flow      = None     # (H, W, 2) flow field
        self.bg_vx      = 0.0     # background (camera) motion
        self.bg_vy      = 0.0

    def update(self, gray, blobs):
        """Compute flow, annotate each blob with independent motion magnitude."""
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            self._flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
        else:
            self._flow = None
        self._prev_gray = gray.copy()

        if self._flow is None:
            for b in blobs:
                b["indep_motion"] = 0.0
            return

        # Background motion = median flow across whole frame
        self.bg_vx = float(np.median(self._flow[:, :, 0]))
        self.bg_vy = float(np.median(self._flow[:, :, 1]))

        for b in blobs:
            bx, by, bw, bh = b["bbox"]
            roi = self._flow[by:by+bh, bx:bx+bw]
            if roi.size > 0:
                bvx = float(np.mean(roi[:, :, 0])) - self.bg_vx
                bvy = float(np.mean(roi[:, :, 1])) - self.bg_vy
            else:
                bvx = bvy = 0.0
            b["indep_motion"] = (bvx**2 + bvy**2) ** 0.5


# ── Blob priority scorer (small feedforward neural network) ───────────────────
class BlobScorer:
    """
    Feedforward neural network (7 → 10 → 1) that scores blob priority.

    Input features (normalised):
      0  size_inv      — 1/(1+area/50); smaller blobs score higher
      1  indep_motion  — optical-flow magnitude independent of background (px/frame)
      2  growth_rate   — area change rate (positive = getting larger)
      3  temp_ratio    — blob mean T / frame max T
      4  distinctness  — (blob T - frame mean T) / frame T std
      5  centering     — 1 - dist_to_centre / max_dist; higher = nearer to centre
      6  has_moved     — EMA of historical independent motion; rewards targets that
                         have been moving even if currently paused

    Weights heavily prioritise:
      - Independent motion (any motion at all is a very strong signal)
      - Targets that move then center and grow (the approach pattern)
      - Small blobs over large background areas
    """

    FEATURE_NAMES = ["size", "motion", "growth", "temp", "distinct",
                     "center", "moved"]

    # Hidden layer: 10 neurons
    # Key insight: size_inv gates motion neurons so large blobs don't benefit
    # from motion as much. Small blobs with any motion dominate.
    W1 = np.array([
        # sz   mot   grw   tmp   dst   ctr   mvd
        [ 3.0,  0.3,  0.2,  0.1,  1.0,  0.0,  0.5],  # small+distinct
        [ 4.0, 10.0,  0.5,  0.0,  0.0,  0.0,  0.0],  # small+motion (size gates)
        [ 3.0,  6.0,  0.0,  0.0,  0.0,  0.0,  2.0],  # small+motion+history
        [ 0.3,  0.5,  5.0,  0.0,  0.3,  0.0,  0.0],  # growth-dominant
        [ 3.0,  4.0,  4.0,  0.0,  0.0,  2.0,  0.0],  # small+motion+growth+centering
        [ 2.0,  3.0,  3.0,  0.0,  0.0,  4.0,  2.0],  # approach: small+move+center+grow
        [ 4.0,  2.0,  2.0,  0.0,  0.5,  1.0,  1.0],  # small+moving+growing
        [-1.0,  0.2,  0.1,  2.0,  1.5,  0.0,  0.0],  # thermal fallback
        [ 3.0,  5.0,  0.0,  0.0,  0.0,  0.0,  5.0],  # small+has-moved (sticky)
        [ 0.5,  1.0,  0.5,  0.3,  0.5,  0.5,  0.3],  # balanced
    ], dtype=np.float32)

    b1 = np.array([0.0, -0.3, 0.0, 0.0, -0.5, -0.5, 0.0, 0.3, -0.3, 0.0],
                  dtype=np.float32)

    W2 = np.array([[1.0, 5.0, 3.0, 2.5, 4.0, 5.0, 2.0, 0.4, 3.5, 0.8]],
                  dtype=np.float32)
    b2 = np.array([0.0], dtype=np.float32)

    @classmethod
    def score(cls, feats):
        """Forward pass → scalar priority score. feats: array of 7 floats."""
        x = np.asarray(feats, dtype=np.float32)
        h = np.maximum(0.0, cls.W1 @ x + cls.b1)
        return float((cls.W2 @ h + cls.b2)[0])

    @classmethod
    def feature_contributions(cls, feats):
        """Per-feature contribution via perturbation."""
        x = np.asarray(feats, dtype=np.float32)
        h = np.maximum(0.0, cls.W1 @ x + cls.b1)
        base = float((cls.W2 @ h + cls.b2)[0])
        contribs = {}
        for i, name in enumerate(cls.FEATURE_NAMES):
            x2 = x.copy()
            x2[i] += 1.0
            h2 = np.maximum(0.0, cls.W1 @ x2 + cls.b1)
            contribs[name] = float((cls.W2 @ h2 + cls.b2)[0]) - base
        return contribs


# ── Hot blob tracker ───────────────────────────────────────────────────────────
class HotBlobTracker:
    """
    Two-state tracker: SEARCHING → LOCKED.

    SEARCHING  Score all blobs using BlobScorer neural network.  Prefers small,
               independently moving, growing blobs.  Must be stable for
               CONFIRM_SECS before lock is acquired.

    LOCKED     Predict next position with smoothed velocity; search within
               an expanding radius around the prediction.  Hold lock through
               brief occlusions (up to MAX_MISS_SECS).  Only breaks lock
               when the object has truly gone.
    """
    CONFIRM_SECS   = 0.5    # pre-lock stability window (longer = fewer false locks)
    MAX_DRIFT_PX   = 20     # max per-frame jump still treated as same blob (SEARCHING)
    SEARCH_RADIUS  = 20     # tighter search radius when locked (was 35)
    VEL_ALPHA      = 0.75   # velocity EMA smoothing (higher = smoother, less jitter)
    MAX_MISS_SECS  = 3.0    # hold lock this long with no detection before giving up
    MIN_TEMP_RATIO = 0.50   # locked candidate must be ≥ this fraction of frame max T
    SIZE_HISTORY_N = 150    # entries of area history for growth-rate estimation
    PREEMPT_RATIO  = 5.0    # break lock only if another blob scores 5× better (was 3×)
    MIN_SCORE      = 200.0  # minimum nn_score to be considered a valid target
    MIN_MOTION     = 0.8    # minimum motion feature to confirm lock (reject static blobs)

    def __init__(self):
        self._state      = "SEARCHING"
        self._history    = deque()   # (time, cx, cy, score) during SEARCHING
        self.confirmed   = None      # last confirmed blob dict
        self.last_blobs  = []        # all scored blobs from most recent frame
        self.sticky_blobs = []       # blobs with 3s persistence for display
        # LOCKED state variables
        self._lx = self._ly = 0.0   # last known centroid
        self._vx = self._vy = 0.0   # smoothed velocity (sensor px/s)
        self._lt         = 0.0      # smoothed temperature
        self._last_seen  = 0.0
        self._lock_since = 0.0
        self._prev_time  = None
        # Motion analysis
        self._motion     = MotionAnalyser()
        # Size history: deque of (time, area) for growth-rate estimation
        self._size_hist  = deque(maxlen=self.SIZE_HISTORY_N)
        # Per-blob motion history (keyed by approximate position bucket)
        self._motion_ema = {}  # (bx//8, by//8) → EMA of indep_motion

    # ── public ────────────────────────────────────────────────────────────────
    def update(self, gray, T):
        now = time.time()
        dt  = (now - self._prev_time) if self._prev_time else 0.033
        dt  = max(dt, 1e-3)
        self._prev_time = now

        blobs = self._find_all_blobs(T)
        self._motion.update(gray, blobs)
        self._score_blobs(blobs, T, now)
        self.last_blobs = sorted(blobs, key=lambda b: b["nn_score"], reverse=True)
        self._update_sticky(now)

        if self._state == "SEARCHING":
            return self._step_searching(blobs, now)
        else:
            return self._step_locked(blobs, now, dt, T)

    def seconds_tracked(self):
        if self._state == "LOCKED":
            return time.time() - self._lock_since
        if len(self._history) >= 2:
            return self._history[-1][0] - self._history[0][0]
        return 0.0

    # ── blob scoring ─────────────────────────────────────────────────────────
    def _score_blobs(self, blobs, T, now):
        """Compute neural-network priority score for each blob."""
        valid = T[~np.isnan(T)]
        if len(valid) == 0:
            for b in blobs:
                b["nn_score"] = 0.0
                b["features"] = {}
                b["contribs"] = {}
            return

        frame_max  = float(valid.max())
        frame_mean = float(valid.mean())
        frame_std  = float(valid.std()) if len(valid) > 1 else 1.0
        frame_std  = max(frame_std, 0.1)
        max_dist   = ((THERMAL_W/2)**2 + (THERMAL_H/2)**2) ** 0.5

        for b in blobs:
            # Hard-suppress oversized blobs
            if b.get("_suppressed") or b["area"] > BLOB_MAX_AREA:
                b["nn_score"] = 0.0
                b["features"] = dict(size=0, motion=0, growth=0, temp=0,
                                     distinct=0, center=0, moved=0)
                b["contribs"] = {n: 0.0 for n in BlobScorer.FEATURE_NAMES}
                continue

            # Steep inverse: area=50 → 0.5, area=200 → 0.2, area=1000 → 0.05
            size_inv     = 1.0 / (1.0 + b["area"] / 25.0)
            # Amplify: raw Farneback values at 256x192 are sub-pixel (0.1–1.5),
            # scale by 5× so the neural net sees meaningful differentiation.
            # Then multiply by size_inv so large blobs' motion is discounted —
            # a small blob moving at 0.5px is a much stronger signal than a
            # 1000px region showing the same flow (likely background/noise).
            raw_motion   = min(b.get("indep_motion", 0.0) * 5.0, 15.0)
            indep_motion = raw_motion * size_inv * 5.0  # size-gated motion
            temp_ratio   = b["t_mean"] / frame_max if frame_max > 0 else 0.0
            distinctness = (b["t_mean"] - frame_mean) / frame_std
            growth_rate  = self._estimate_growth(b, now)

            # Centering: how close to image centre (0=edge, 1=dead centre)
            cdist = ((b["cx"] - THERMAL_W/2)**2 + (b["cy"] - THERMAL_H/2)**2) ** 0.5
            centering = 1.0 - min(cdist / max_dist, 1.0)

            # Has-moved: EMA of historical motion for this spatial bucket
            bkey = (b["cx"] // 8, b["cy"] // 8)
            prev_ema = self._motion_ema.get(bkey, 0.0)
            has_moved = 0.7 * prev_ema + 0.3 * indep_motion
            self._motion_ema[bkey] = has_moved

            feats = [size_inv, indep_motion, growth_rate, temp_ratio,
                     distinctness, centering, has_moved]

            b["nn_score"] = BlobScorer.score(feats)
            b["features"] = dict(size=size_inv, motion=indep_motion,
                                 growth=growth_rate, temp=temp_ratio,
                                 distinct=distinctness, center=centering,
                                 moved=has_moved)
            b["contribs"] = BlobScorer.feature_contributions(feats)

        # Decay stale motion_ema entries periodically
        if len(self._motion_ema) > 200:
            self._motion_ema = {k: v for k, v in self._motion_ema.items()
                                if v > 0.01}

    def _update_sticky(self, now):
        """Merge current blobs into sticky list; expire after PANEL_STICKY seconds."""
        # Update existing sticky entries with fresh blob data
        live_keys = set()
        for b in self.last_blobs:
            bkey = (b["cx"] // 6, b["cy"] // 6)
            live_keys.add(bkey)
            # Find matching sticky entry
            matched = False
            for s in self.sticky_blobs:
                sk = (s["blob"]["cx"] // 6, s["blob"]["cy"] // 6)
                if sk == bkey:
                    s["blob"] = b
                    s["last_seen"] = now
                    matched = True
                    break
            if not matched:
                self.sticky_blobs.append({"blob": b, "last_seen": now})

        # Expire old entries
        self.sticky_blobs = [s for s in self.sticky_blobs
                             if now - s["last_seen"] < PANEL_STICKY]
        # Sort by score, keep top PANEL_MAX
        self.sticky_blobs.sort(key=lambda s: s["blob"].get("nn_score", 0),
                               reverse=True)
        if len(self.sticky_blobs) > PANEL_MAX:
            self.sticky_blobs = self.sticky_blobs[:PANEL_MAX]

    def _estimate_growth(self, blob, now):
        """Estimate area growth rate for a blob near the current position."""
        cx, cy = blob["cx"], blob["cy"]
        # Find recent size_hist entries near this blob
        relevant = [(t, a) for t, x, y, a in self._size_hist
                    if ((x - cx)**2 + (y - cy)**2) ** 0.5 < 30]
        if len(relevant) < 3:
            return 0.0
        # Linear regression: area vs time
        times  = np.array([t for t, a in relevant])
        areas  = np.array([a for t, a in relevant], dtype=np.float32)
        dt     = times - times[0]
        dt_max = dt[-1]
        if dt_max < 0.1:
            return 0.0
        # slope via least-squares
        n      = len(dt)
        sum_t  = dt.sum()
        sum_a  = areas.sum()
        sum_ta = (dt * areas).sum()
        sum_tt = (dt * dt).sum()
        denom  = n * sum_tt - sum_t * sum_t
        if abs(denom) < 1e-9:
            return 0.0
        slope  = (n * sum_ta - sum_t * sum_a) / denom
        # Normalise: rate as fraction of mean area per second
        mean_a = areas.mean()
        return float(slope / max(mean_a, 1.0))

    # ── private ───────────────────────────────────────────────────────────────
    def _step_searching(self, blobs, now):
        if not blobs:
            self._history.clear()
            self.confirmed = None
            return None

        best = max(blobs, key=lambda b: b["nn_score"])

        # Record size history for growth-rate estimation
        for b in blobs:
            self._size_hist.append((now, b["cx"], b["cy"], b["area"]))

        # Reject if best blob doesn't meet minimum score threshold —
        # no target is better than a false positive
        if best["nn_score"] < self.MIN_SCORE:
            self._history.clear()
            self.confirmed = None
            return None

        # Reject if best blob has no meaningful independent motion —
        # a static hot region is not a target, no matter how hot
        best_motion = best.get("features", {}).get("motion", 0)
        if best_motion < self.MIN_MOTION:
            self._history.clear()
            self.confirmed = None
            return None

        cx, cy = best["cx"], best["cy"]

        # Expire old entries
        cutoff = now - (self.CONFIRM_SECS + 0.5)
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        # Reset if best blob jumped too far — but allow larger drift for
        # high-scoring blobs (small moving targets can be noisy in position)
        if self._history:
            px, py = self._history[-1][1], self._history[-1][2]
            drift = ((cx - px)**2 + (cy - py)**2)**0.5
            drift_limit = self.MAX_DRIFT_PX
            if best["nn_score"] > 500:
                drift_limit = self.MAX_DRIFT_PX * 2.5  # more tolerance for good targets
            if drift > drift_limit:
                self._history.clear()

        self._history.append((now, cx, cy, best["nn_score"]))

        if (now - self._history[0][0]) >= self.CONFIRM_SECS:
            # Acquire lock
            self._state      = "LOCKED"
            self._lx, self._ly = float(cx), float(cy)
            self._vx = self._vy = 0.0
            self._lt         = best["t_mean"]
            self._last_seen  = now
            self._lock_since = now
            self._history.clear()
            self.confirmed   = best

        # Return None until locked; caller shows progress bar via seconds_tracked()
        return self.confirmed

    def _step_locked(self, blobs, now, dt, T):
        time_since = now - self._last_seen

        # Record size history for all visible blobs
        for b in blobs:
            self._size_hist.append((now, b["cx"], b["cy"], b["area"]))

        # ── Preemption: if a much better blob exists anywhere, break lock ─
        if blobs and self.confirmed:
            global_best = max(blobs, key=lambda b: b["nn_score"])
            cur_score   = self.confirmed.get("nn_score", 0)
            if (global_best["nn_score"] > cur_score * self.PREEMPT_RATIO
                    and global_best["nn_score"] > 100):
                # Immediately switch to the better target
                self._state    = "SEARCHING"
                self._history.clear()
                self.confirmed = None
                return self._step_searching(blobs, now)

        # Predict position based on velocity
        pred_x = self._lx + self._vx * dt
        pred_y = self._ly + self._vy * dt

        # If predicted position is outside the frame, target has exited — release
        margin = 5
        if (pred_x < -margin or pred_x > THERMAL_W + margin or
                pred_y < -margin or pred_y > THERMAL_H + margin):
            self._state    = "SEARCHING"
            self._history.clear()
            self.confirmed = None
            return None

        # Search radius grows with velocity uncertainty over time
        radius = self.SEARCH_RADIUS + (abs(self._vx) + abs(self._vy)) * time_since

        # Minimum plausible temperature for the locked object
        valid_T = T[~np.isnan(T)]
        min_t = float(valid_T.max()) * self.MIN_TEMP_RATIO if len(valid_T) else 0.0

        # Score candidates: blend proximity with neural score
        best_blob, best_score = None, -1e9
        for b in blobs:
            dist = ((b["cx"] - pred_x)**2 + (b["cy"] - pred_y)**2)**0.5
            if dist > radius or b["t_mean"] < min_t:
                continue
            if b["nn_score"] < self.MIN_SCORE:
                continue  # don't re-lock onto noise
            # Strongly prefer blobs near the predicted position —
            # penalise distance heavily to prevent jumping to nearby hot spots
            score = b["nn_score"] * 0.5 - dist * 1.5 + b["t_mean"] * 0.1
            if score > best_score:
                best_score = score
                best_blob  = b

        if best_blob is not None:
            # Update velocity with EMA
            raw_vx = (best_blob["cx"] - self._lx) / dt
            raw_vy = (best_blob["cy"] - self._ly) / dt
            self._vx = self.VEL_ALPHA * self._vx + (1 - self.VEL_ALPHA) * raw_vx
            self._vy = self.VEL_ALPHA * self._vy + (1 - self.VEL_ALPHA) * raw_vy
            self._lx = float(best_blob["cx"])
            self._ly = float(best_blob["cy"])
            self._lt = 0.8 * self._lt + 0.2 * best_blob["t_mean"]
            self._last_seen = now
            self.confirmed  = best_blob

            # Release lock if target has become stationary — a real target
            # should have meaningful motion or it's just a warm pixel
            speed = (self._vx**2 + self._vy**2) ** 0.5
            blob_motion = best_blob.get("features", {}).get("motion", 0)
            lock_age = now - self._lock_since
            if lock_age > 2.0 and speed < 0.5 and blob_motion < 0.3:
                self._state    = "SEARCHING"
                self._history.clear()
                self.confirmed = None
                return None

        elif time_since > self.MAX_MISS_SECS:
            # Truly lost — back to SEARCHING
            self._state    = "SEARCHING"
            self.confirmed = None
            return None

        else:
            # Coast: advance predicted position using established trajectory
            self._lx = pred_x
            self._ly = pred_y
            # If coasting position exits the frame, target is gone
            if (self._lx < -margin or self._lx > THERMAL_W + margin or
                    self._ly < -margin or self._ly > THERMAL_H + margin):
                self._state    = "SEARCHING"
                self._history.clear()
                self.confirmed = None
                return None

        return self.confirmed

    def _find_all_blobs(self, T):
        """
        Multi-level blob finder.  First tries a high threshold (mean + 3*std)
        to isolate genuinely hot small targets.  Falls back to the percentile
        threshold if nothing is found.  Large blobs (> BLOB_MAX_AREA) are
        split by re-thresholding their interior at a higher level to extract
        any bright peak embedded within them.
        """
        valid = ~np.isnan(T)
        if valid.sum() < BLOB_MIN_AREA + 1:
            return []

        T_valid = T[valid]
        frame_mean = float(T_valid.mean())
        frame_std  = float(T_valid.std()) if len(T_valid) > 1 else 1.0

        # Two-level approach:
        # Level 1: percentile threshold → catches everything warm
        pct_thresh  = float(np.percentile(T_valid, BLOB_THRESHOLD_PCT))
        blobs_pct   = self._extract_blobs(T, valid, pct_thresh, open_morph=True)

        # Level 2: statistical outlier threshold (mean + 2.5σ) → isolates hot peaks
        stat_thresh = frame_mean + 2.5 * frame_std
        blobs_stat  = self._extract_blobs(T, valid, stat_thresh, open_morph=False)

        # Prefer stat blobs if they contain any small targets
        small_stat = [b for b in blobs_stat if BLOB_MIN_AREA < b["area"] <= BLOB_MAX_AREA]
        small_pct  = [b for b in blobs_pct  if BLOB_MIN_AREA < b["area"] <= BLOB_MAX_AREA]

        if small_stat:
            blobs = blobs_stat
        elif small_pct:
            blobs = blobs_pct
        else:
            # No small blobs at either level — try even higher to split big blobs
            blobs = blobs_pct
            if blobs and all(b["area"] > BLOB_MAX_AREA for b in blobs):
                for extra_sigma in [3.5, 4.5, 5.5]:
                    higher = frame_mean + extra_sigma * frame_std
                    split = self._extract_blobs(T, valid, higher, open_morph=False)
                    small_split = [b for b in split if b["area"] <= BLOB_MAX_AREA]
                    if small_split:
                        blobs = split
                        break

        # Suppress oversized blobs
        for b in blobs:
            if b["area"] > BLOB_MAX_AREA:
                b["_suppressed"] = True
        return blobs

    def _extract_blobs(self, T, valid, thresh_T, open_morph=True):
        """Extract connected components above thresh_T."""
        hot_mask = (valid & (T >= thresh_T)).astype(np.uint8) * 255
        if open_morph:
            kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            hot_mask, connectivity=8)

        blobs = []
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area <= BLOB_MIN_AREA:
                continue
            blob_mask = labels == lbl
            valid_t   = T[blob_mask & valid]
            if not len(valid_t):
                continue
            blobs.append(dict(
                cx    = int(centroids[lbl][0]),
                cy    = int(centroids[lbl][1]),
                t_mean= float(valid_t.mean()),
                t_max = float(valid_t.max()),
                bbox  = (stats[lbl, cv2.CC_STAT_LEFT],  stats[lbl, cv2.CC_STAT_TOP],
                         stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT]),
                area  = area,
            ))
        return blobs


# ── Drawing helpers ────────────────────────────────────────────────────────────
def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except Exception:
        return None


def get_usb_bus():
    try:
        out = subprocess.check_output(["lsusb"], text=True)
        for line in out.splitlines():
            if "2bdf:0102" in line or "HikCamera" in line:
                p = line.split()
                return f"Bus {p[1]} Dev {p[3].rstrip(':')}"
    except Exception:
        pass
    return "USB"


def draw_text(img, text, pos, scale=0.55, thickness=1, color=OVERLAY_COLOR, shadow=True):
    x, y = pos
    if shadow:
        cv2.putText(img, text, (x+1, y+1), FONT, scale, SHADOW_COLOR, thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_graph(canvas, history, now, gx, gy, gw, gh, label, axis_labels,
               markers=None):
    """
    Draw a time-series graph at (gx, gy) with size (gw, gh).
    history: list of (timestamp, value).
    axis_labels: (neg_label, pos_label) e.g. ("Left", "Right").
    markers: iterable of timestamps to draw as vertical marker lines.
    """
    # Background + border
    cv2.rectangle(canvas, (gx, gy), (gx+gw, gy+gh), (45, 45, 45), -1)
    cv2.rectangle(canvas, (gx, gy), (gx+gw, gy+gh), (80, 80, 80), 1)

    # Centre line (zero)
    cy = gy + gh // 2
    cv2.line(canvas, (gx+1, cy), (gx+gw-1, cy), (70, 70, 70), 1)

    # Axis labels
    draw_text(canvas, axis_labels[0], (gx + 2, cy + 12), scale=0.3, shadow=False,
              color=(100, 100, 100))
    tw, _ = cv2.getTextSize(axis_labels[1], FONT, 0.3, 1)[0]
    draw_text(canvas, axis_labels[1], (gx + gw - tw - 2, cy + 12), scale=0.3,
              shadow=False, color=(100, 100, 100))

    # Title
    draw_text(canvas, label, (gx + 2, gy + 12), scale=0.35, shadow=False,
              color=(180, 180, 180))

    # Target-change markers — vertical dashed lines
    t_start = now - GRAPH_SECS
    if markers:
        for mt in markers:
            frac = (mt - t_start) / GRAPH_SECS
            if frac < 0 or frac > 1:
                continue
            mx = gx + int(frac * (gw - 2)) + 1
            mcol = (0, 180, 0)
            cv2.line(canvas, (mx, gy + 1), (mx, gy + gh - 1), mcol, 1)
            cv2.drawMarker(canvas, (mx, gy + 4), mcol, cv2.MARKER_TRIANGLE_DOWN,
                           8, 1, cv2.LINE_AA)

    if len(history) < 2:
        return

    # Find range — symmetric around zero, minimum ±10
    vals = [v for _, v in history]
    abs_max = max(abs(min(vals)), abs(max(vals)), 10)

    # Scale labels
    top_s = f"+{abs_max:.0f}"
    bot_s = f"-{abs_max:.0f}"
    tw_top, _ = cv2.getTextSize(top_s, FONT, 0.28, 1)[0]
    draw_text(canvas, top_s, (gx + gw - tw_top - 2, gy + 11), scale=0.28,
              shadow=False, color=(90, 90, 90))
    tw_bot, _ = cv2.getTextSize(bot_s, FONT, 0.28, 1)[0]
    draw_text(canvas, bot_s, (gx + gw - tw_bot - 2, gy + gh - 3), scale=0.28,
              shadow=False, color=(90, 90, 90))

    # Plot points
    t_start = now - GRAPH_SECS
    pts = []
    for t, v in history:
        frac = (t - t_start) / GRAPH_SECS
        if frac < 0:
            continue
        px = gx + int(frac * (gw - 2)) + 1
        py = cy - int((v / abs_max) * (gh // 2 - 2))
        py = max(gy + 1, min(gy + gh - 2, py))
        pts.append((px, py))

    if len(pts) >= 2:
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False,
                      (0, 200, 255), 1, cv2.LINE_AA)

    # Current value at the right end
    if pts:
        last_val = vals[-1]
        cur_s = f"{last_val:+.0f}"
        draw_text(canvas, cur_s, (gx + gw + 3, pts[-1][1] + 4), scale=0.35,
                  shadow=False, color=(0, 200, 255))


def build_colorbar(height, colormap, width=22):
    bar = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    return cv2.applyColorMap(np.repeat(bar, width, axis=1), colormap)


def draw_hot_blob(canvas, blob, tracker, disp_w, disp_h, ox, oy, show_temps=False):
    """Draw blob overlay. ox, oy = pixel offset of thermal image within canvas."""
    S  = SCALE
    cx = ox + blob["cx"] * S
    cy = oy + blob["cy"] * S
    bx, by, bw, bh = blob["bbox"]
    rx1, ry1 = ox + bx*S, oy + by*S
    rx2, ry2 = ox + (bx+bw)*S, oy + (by+bh)*S

    img_cx, img_cy = ox + disp_w//2, oy + disp_h//2

    # Signed offsets: X positive = right of centre, Y positive = above centre
    dx =   blob["cx"] - THERMAL_W // 2
    dy = -(blob["cy"] - THERMAL_H // 2)

    t_mean = blob["t_mean"]
    t_max  = blob["t_max"]
    secs   = tracker.seconds_tracked()

    pulse = abs(np.sin(time.time() * 4)) * 0.4 + 0.6
    hcol  = (0, 0, int(255 * pulse))

    cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), hcol, 2, cv2.LINE_AA)
    cv2.drawMarker(canvas, (cx, cy), hcol, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

    vec  = np.array([cx - img_cx, cy - img_cy], dtype=float)
    dist = np.linalg.norm(vec)
    if dist > 1:
        tip = (int(img_cx + vec[0]*max(0, dist-14)/dist),
               int(img_cy + vec[1]*max(0, dist-14)/dist))
        cv2.arrowedLine(canvas, (img_cx, img_cy), tip,
                        (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.06)

    if show_temps:
        if not np.isnan(t_max):
            temp_label = f"{t_max:.1f}C"
        elif not np.isnan(t_mean):
            temp_label = f"~{t_mean:.1f}C"
        else:
            temp_label = "HOT"
        label_y = max(ry1 - 8, oy + 14)
        draw_text(canvas, temp_label, (rx1, label_y), scale=0.65, thickness=2, color=hcol)

    coord_str = f"X:{dx:+d} Y:{dy:+d}  [{secs:.1f}s]"
    label_y2 = min(ry2 + 18, oy + disp_h - 6)
    draw_text(canvas, coord_str, (rx1, label_y2), scale=0.48, color=hcol)


def draw_blob_pending(canvas, tracker, ox, oy):
    secs = tracker.seconds_tracked()
    if secs <= 0:
        return
    pct   = min(secs / HotBlobTracker.CONFIRM_SECS, 1.0)
    bar_x, bar_y = ox + 8, oy - 18
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+100, bar_y+8), (60,60,60), -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+int(100*pct), bar_y+8), (0,200,255), -1)
    draw_text(canvas, "locking...", (bar_x, bar_y-4), scale=0.4, color=(0,200,255))


def draw_dotted_rect(img, pt1, pt2, color, thickness=1, gap=6):
    """Draw a dotted rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    for edge in [((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
                 ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1))]:
        (sx, sy), (ex, ey) = edge
        length = max(abs(ex-sx), abs(ey-sy))
        if length == 0:
            continue
        dx = (ex - sx) / length
        dy = (ey - sy) / length
        i = 0
        while i < length:
            seg = min(gap // 2 + 1, length - i)
            px1 = int(sx + dx * i)
            py1 = int(sy + dy * i)
            px2 = int(sx + dx * min(i + seg, length))
            py2 = int(sy + dy * min(i + seg, length))
            cv2.line(img, (px1, py1), (px2, py2), color, thickness)
            i += gap


def draw_secondary_blobs(canvas, blobs, confirmed, ox, oy):
    """Draw blue dotted boxes around non-primary candidate blobs."""
    S = SCALE
    for b in blobs:
        if confirmed and b["cx"] == confirmed["cx"] and b["cy"] == confirmed["cy"]:
            continue
        bx, by, bw, bh = b["bbox"]
        rx1, ry1 = ox + bx*S, oy + by*S
        rx2, ry2 = ox + (bx+bw)*S, oy + (by+bh)*S
        draw_dotted_rect(canvas, (rx1, ry1), (rx2, ry2), (200, 150, 0), 1)


def _rejection_reason(blob, best_blob):
    """Return a short string explaining why this blob is not the primary target."""
    if blob is best_blob:
        return ""
    f = blob.get("features", {})
    best_f = best_blob.get("features", {})
    c = blob.get("contribs", {})

    reasons = []
    if f.get("size", 1) < 0.3 and best_f.get("size", 0) > 0.5:
        reasons.append("too large")
    if f.get("motion", 0) < 0.3 and best_f.get("motion", 0) > 0.5:
        reasons.append("static")
    if f.get("moved", 0) < 0.1 and best_f.get("moved", 0) > 0.3:
        reasons.append("no motion history")
    if f.get("growth", 0) < 0.05 and best_f.get("growth", 0) > 0.1:
        reasons.append("not growing")
    if f.get("center", 0) < 0.3 and best_f.get("center", 0) > 0.6:
        reasons.append("off-centre")
    if f.get("distinct", 0) < best_f.get("distinct", 0) * 0.5:
        reasons.append("low contrast")

    if not reasons:
        if c and best_blob.get("contribs"):
            worst_feat = min(c, key=lambda k: c[k] - best_blob["contribs"].get(k, 0))
            label_map = {"size": "too large", "motion": "less motion",
                         "growth": "less growth", "temp": "cooler",
                         "distinct": "less distinct", "center": "off-centre",
                         "moved": "no motion history"}
            reasons.append(label_map.get(worst_feat, "lower score"))
    return "; ".join(reasons[:2])


def draw_candidate_panel(canvas, sticky_blobs, confirmed, thermal_img,
                         px, py, pw, ph, now):
    """
    Draw top-N candidate panel at position (px, py) with width pw, height ph.
    sticky_blobs: list of {"blob": dict, "last_seen": float}.
    """
    if not sticky_blobs:
        draw_text(canvas, "No candidates", (px + 4, py + 16), scale=0.35,
                  shadow=False, color=(100, 100, 100))
        return

    draw_text(canvas, "Candidates", (px + 4, py + 14), scale=0.38,
              shadow=False, color=(180, 180, 180))

    entries = sticky_blobs[:PANEL_MAX]
    all_blobs = [s["blob"] for s in entries]
    best_blob = all_blobs[0] if all_blobs else None

    scores = [b["nn_score"] for b in all_blobs]
    shifted = [s - max(scores) for s in scores]
    exps = [np.exp(s * 0.3) for s in shifted]
    exp_sum = sum(exps)
    probs = [e / exp_sum for e in exps] if exp_sum > 0 else [1.0 / len(exps)] * len(exps)

    th   = PANEL_THUMB
    slot = (ph - 20) // PANEL_MAX
    slot = min(slot, th + 24)
    S    = SCALE

    for i, entry in enumerate(entries):
        b    = entry["blob"]
        age  = now - entry["last_seen"]
        fade = max(0.3, 1.0 - age / PANEL_STICKY) if age > 0.5 else 1.0

        sy = py + 20 + i * slot
        is_primary = (confirmed is not None and
                      b["cx"] == confirmed["cx"] and b["cy"] == confirmed["cy"])

        bx, by, bw, bh = b["bbox"]
        pad = max(bw, bh) // 2 + 2
        crop_x1 = max(0, (bx - pad) * S)
        crop_y1 = max(0, (by - pad) * S)
        crop_x2 = min(thermal_img.shape[1], (bx + bw + pad) * S)
        crop_y2 = min(thermal_img.shape[0], (by + bh + pad) * S)

        crop = thermal_img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size > 0:
            thumb = cv2.resize(crop, (th, th), interpolation=cv2.INTER_LINEAR)
            if fade < 0.9:
                thumb = (thumb.astype(np.float32) * fade).astype(np.uint8)
        else:
            thumb = np.full((th, th, 3), 40, dtype=np.uint8)

        tx1 = px + 4
        ty1 = sy
        canvas[ty1:ty1+th, tx1:tx1+th] = thumb

        bcol = (0, 220, 0) if is_primary else (80, 80, 80)
        cv2.rectangle(canvas, (tx1-1, ty1-1), (tx1+th, ty1+th), bcol, 1)

        bar_x = tx1 + th + 4
        bar_w = pw - th - 16
        bar_h = 6

        cv2.rectangle(canvas, (bar_x, ty1+2), (bar_x + bar_w, ty1+2+bar_h),
                      (50, 50, 50), -1)
        fill_w = max(1, int(bar_w * probs[i]))
        bar_col = (0, 200, 0) if is_primary else (0, 150, 200)
        cv2.rectangle(canvas, (bar_x, ty1+2), (bar_x + fill_w, ty1+2+bar_h),
                      bar_col, -1)

        pct_s = f"{probs[i]*100:.0f}%"
        txt_c = tuple(int(c * fade) for c in (180, 180, 180))
        draw_text(canvas, pct_s, (bar_x + bar_w + 2, ty1 + 10), scale=0.28,
                  shadow=False, color=txt_c)

        f = b.get("features", {})
        motion_s = f"m:{f.get('motion',0):.1f}"
        growth_s = f"g:{f.get('growth',0):+.2f}"
        moved_s  = f"h:{f.get('moved',0):.1f}"
        area_s   = f"a:{b['area']}"
        info = f"{area_s} {motion_s} {growth_s} {moved_s}"
        draw_text(canvas, info, (bar_x, ty1 + 20), scale=0.27,
                  shadow=False, color=tuple(int(c * fade) for c in (140, 140, 140)))

        if not is_primary and best_blob:
            reason = _rejection_reason(b, best_blob)
            if reason:
                draw_text(canvas, reason, (bar_x, ty1 + 32), scale=0.26,
                          shadow=False, color=tuple(int(c * fade) for c in (100, 100, 180)))
        elif is_primary:
            draw_text(canvas, "PRIMARY", (bar_x, ty1 + 32), scale=0.28,
                      shadow=False, color=(0, 220, 0))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Opening {VENDOR} {MODEL}  S/N {SERIAL}  (build {BUILD_TIME})")
    print("Keys: [q] quit  [f] fullscreen  [t] toggle temps  [r] record  [s] screenshot")
    print("      [o/p] open recording  [D] diagnostic dump")
    print("      [K] calibrate HOT  — fill frame with hot kettle, press K (= 80°C)")
    print("      [A] calibrate AMBIENT — point at room-temp scene, press A (= 22°C)")
    print("      [0] reset calibration  (K alone is usually enough)")
    print("  Note: K uses frame 99th-percentile u16, A uses frame median — no crosshair aiming needed")

    load_calibration()
    cam     = ThermalCamera(DEVICE)
    usb_bus = get_usb_bus()

    WIN = "Thermal Camera - HIK HikCamera"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    init_w = MARGIN_L + THERMAL_W * SCALE + MARGIN_R
    init_h = MARGIN_TOP + THERMAL_H * SCALE + MARGIN_BOT
    cv2.resizeWindow(WIN, init_w, init_h)

    tracker = HotBlobTracker()

    fps_ring   = []
    prev_time  = time.time()
    frame_num  = 0
    rec_writers = None   # (raw_writer, ann_writer) tuple while recording
    rec_start   = None   # datetime when recording began
    rec_folder  = None   # recording output folder
    pos_history = deque()   # (timestamp, dx, dy) for position graphs
    tgt_changes = deque()   # timestamps when target changed (for graph markers)
    prev_locked = False     # was tracker locked last frame?
    show_temps  = False     # toggle with T key

    while True:
        gray, T, u16_raw = cam.read(timeout=1.0)
        if gray is None:
            print("Frame timeout — retrying...")
            continue

        # Apply user calibration to temperature array
        T = apply_cal(T)

        now = time.time()
        fps_ring.append(1.0 / max(now - prev_time, 1e-6))
        if len(fps_ring) > 30:
            fps_ring.pop(0)
        prev_time = now
        fps = np.mean(fps_ring)
        frame_num += 1

        # ── Visual thermal display: greyscale + colour for top hot pixels ─────
        mn, mx = int(gray.min()), int(gray.max())
        if mx > mn:
            norm = ((gray.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            norm = gray.copy()

        disp_w = THERMAL_W * SCALE
        disp_h = THERMAL_H * SCALE

        # Grey base
        gray_bgr = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        # Inferno colour layer for hot regions
        colored  = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        # Adaptive colour mask: only colour genuinely hot outliers.
        # Use whichever is more selective: percentile or mean+3σ.
        # If very few pixels are true outliers, only they get colour.
        T_valid_disp = T[~np.isnan(T)]
        if len(T_valid_disp):
            pct_thresh  = float(np.percentile(T_valid_disp, HOT_DISPLAY_PCT))
            stat_thresh = float(T_valid_disp.mean() + 3.0 * max(T_valid_disp.std(), 0.1))
            hot_thresh  = max(pct_thresh, stat_thresh)
        else:
            hot_thresh = 999.0
        hot_mask = ((~np.isnan(T)) & (T >= hot_thresh)).astype(np.uint8)
        # Combine at sensor resolution, then upscale once
        combined = np.where(hot_mask[:, :, np.newaxis], colored, gray_bgr)
        thermal  = cv2.resize(combined, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        raw_frame = thermal.copy()   # no-overlay copy for recording

        # ── Build canvas with margins ─────────────────────────────────────────
        canvas_w = MARGIN_L + disp_w + MARGIN_R
        canvas_h = MARGIN_TOP + disp_h + MARGIN_BOT
        ox, oy   = MARGIN_L, MARGIN_TOP   # thermal image origin in canvas

        canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)
        canvas[oy:oy+disp_h, ox:ox+disp_w] = thermal

        # ── Hot blob (drawn on top of thermal region) ─────────────────────────
        confirmed = tracker.update(gray, T)

        # Draw blue dotted boxes around non-primary candidates
        draw_secondary_blobs(canvas, tracker.last_blobs, confirmed, ox, oy)

        if confirmed:
            draw_hot_blob(canvas, confirmed, tracker, disp_w, disp_h, ox, oy, show_temps)
        elif tracker.seconds_tracked() > 0:
            draw_blob_pending(canvas, tracker, ox, oy)

        # ── Candidate panel (right side) ──────────────────────────────────────
        panel_x = ox + disp_w + 4
        panel_y = oy
        panel_w = MARGIN_R - 8
        panel_h = disp_h
        draw_candidate_panel(canvas, tracker.sticky_blobs, confirmed,
                             thermal, panel_x, panel_y, panel_w, panel_h, now)

        # Image-centre crosshair (temperature label only if valid)
        img_cx, img_cy = ox + disp_w//2, oy + disp_h//2
        cv2.drawMarker(canvas, (img_cx, img_cy), (0,255,0),
                       cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)
        if show_temps:
            ctr_T = T[THERMAL_H//2, THERMAL_W//2]
            if not np.isnan(ctr_T):
                draw_text(canvas, f"{ctr_T:.1f}C", (img_cx+12, img_cy-6),
                          scale=0.5, color=(0,255,0), shadow=True)

        # ── Target change detection ───────────────────────────────────────────
        is_locked = tracker._state == "LOCKED"
        if is_locked and not prev_locked:
            tgt_changes.append(now)
        prev_locked = is_locked
        # Trim target-change markers to graph window
        cutoff = now - GRAPH_SECS - 1.0
        while tgt_changes and tgt_changes[0] < cutoff:
            tgt_changes.popleft()

        # ── Position history ───────────────────────────────────────────────────
        if confirmed:
            dx_blob =   confirmed["cx"] - THERMAL_W // 2
            dy_blob = -(confirmed["cy"] - THERMAL_H // 2)
            pos_history.append((now, dx_blob, dy_blob))
        while pos_history and pos_history[0][0] < cutoff:
            pos_history.popleft()

        # ── Position graphs (below image, above text) ─────────────────────────
        graph_w  = disp_w - 40   # leave room for current-value label on right
        graph_x  = ox
        graph_y1 = oy + disp_h + 6
        graph_y2 = graph_y1 + GRAPH_H + GRAPH_GAP

        lr_hist = [(t, dx) for t, dx, dy in pos_history]
        ud_hist = [(t, dy) for t, dx, dy in pos_history]
        draw_graph(canvas, lr_hist, now, graph_x, graph_y1, graph_w, GRAPH_H,
                   "X (Left/Right)", ("Left", "Right"), markers=tgt_changes)
        draw_graph(canvas, ud_hist, now, graph_x, graph_y2, graph_w, GRAPH_H,
                   "Y (Up/Down)", ("Down", "Up"), markers=tgt_changes)

        # ── Scene temperature summary ─────────────────────────────────────────
        if show_temps:
            T_valid = T[~np.isnan(T)]
            if len(T_valid):
                scene_str = f"Scene: {T_valid.min():.1f} / {T_valid.mean():.1f} / {T_valid.max():.1f} C"
            else:
                scene_str = "Scene: no temp data"
        else:
            scene_str = ""

        # ── HUD — all text in margins, nothing over thermal image ─────────────
        dt     = datetime.now()
        cpu_t  = get_cpu_temp()
        cpu_s  = f"CPU {cpu_t:.1f}C" if cpu_t else ""

        # Top margin — left: FPS + datetime
        top_left = f"FPS: {fps:5.1f}   {dt.strftime('%Y-%m-%d')}  {dt.strftime('%H:%M:%S.%f')[:-3]}"
        draw_text(canvas, top_left, (ox, MARGIN_TOP - 8), scale=0.5, shadow=False)

        # Top margin — right: device info
        top_right = f"{VENDOR} {MODEL}  S/N:{SERIAL}  {usb_bus}"
        tw, _ = cv2.getTextSize(top_right, FONT, 0.5, 1)[0]
        draw_text(canvas, top_right, (ox + disp_w - tw, MARGIN_TOP - 8), scale=0.5, shadow=False)

        # Recording indicator — top margin, far right
        if rec_writers is not None:
            rec_secs = (dt - rec_start).total_seconds()
            rec_label = f"REC {rec_secs:.0f}s"
            pulse = 0.5 + 0.5 * np.sin(rec_secs * np.pi * 2)
            dot_col = (0, 0, int(180 + 75 * pulse))
            cv2.circle(canvas, (canvas_w - 14, 14), 7, dot_col, -1)
            draw_text(canvas, rec_label, (canvas_w - 82, 20), scale=0.5,
                      color=(0, 0, 255), shadow=False)

        # Bottom margin — row 1 (below graphs)
        bot_y1 = graph_y2 + GRAPH_H + 18
        if scene_str:
            draw_text(canvas, scene_str, (ox, bot_y1), scale=0.5, shadow=False)
        cal_s = cal_status_str()
        tw, _ = cv2.getTextSize(cal_s, FONT, 0.5, 1)[0]
        draw_text(canvas, cal_s, (ox + disp_w - tw, bot_y1), scale=0.5, shadow=False)

        # Bottom margin — row 2
        bot_y2 = bot_y1 + 22
        info_line = f"Thermal {THERMAL_W}x{THERMAL_H} @ 25fps   Hot top {100 - HOT_DISPLAY_PCT}%   Frame: {frame_num}"
        draw_text(canvas, info_line, (ox, bot_y2), scale=0.5, shadow=False)
        if cpu_s:
            tw, _ = cv2.getTextSize(cpu_s, FONT, 0.5, 1)[0]
            draw_text(canvas, cpu_s, (ox + disp_w - tw, bot_y2), scale=0.5, shadow=False)

        # Bottom margin — row 3
        bot_y3 = bot_y2 + 22
        build_s = f"Build: {BUILD_TIME}"
        draw_text(canvas, build_s, (ox, bot_y3), scale=0.45, shadow=False,
                  color=(140, 140, 140))

        # Bottom margin — row 4: pixel stats
        bot_y4 = bot_y3 + 20
        pix_s = f"px min:{mn}  max:{mx}"
        draw_text(canvas, pix_s, (ox, bot_y4), scale=0.4, shadow=False,
                  color=(120, 120, 120))

        cv2.imshow(WIN, canvas)

        # ── Record frames ─────────────────────────────────────────────────────
        if rec_writers is not None:
            rec_writers[0].write(raw_frame)
            rec_writers[1].write(canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f') or key == ord('F'):
            fs = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fs != cv2.WINDOW_FULLSCREEN else cv2.WINDOW_NORMAL)
        elif key == ord('t') or key == ord('T'):
            show_temps = not show_temps
            print(f"  Temperatures: {'shown' if show_temps else 'hidden'}")
        elif key == ord('r') or key == ord('R'):
            if rec_writers is None:
                ts      = dt.strftime('%Y-%m-%d_%H-%M-%S')
                bld     = BUILD_TIME.replace(' ', '_').replace(':', '-')
                rec_dir = os.path.join(os.getcwd(), "thermal_recordings",
                                       f"run_{ts}")
                os.makedirs(rec_dir, exist_ok=True)
                fourcc  = cv2.VideoWriter_fourcc(*'MJPG')
                raw_sz  = (disp_w, disp_h)
                ann_sz  = (canvas_w, canvas_h)
                raw_name = f"raw_{ts}_build-{bld}.avi"
                ann_name = f"annotated_{ts}_build-{bld}.avi"
                raw_w   = cv2.VideoWriter(os.path.join(rec_dir, raw_name),
                                          fourcc, 25.0, raw_sz)
                ann_w   = cv2.VideoWriter(os.path.join(rec_dir, ann_name),
                                          fourcc, 25.0, ann_sz)
                rec_writers = (raw_w, ann_w)
                rec_start   = dt
                rec_folder  = rec_dir
                print(f"  [R] Recording started: {rec_dir}/")
            else:
                rec_writers[0].release()
                rec_writers[1].release()
                dur = (dt - rec_start).total_seconds()
                print(f"  [R] Recording stopped ({dur:.1f}s) → {rec_folder}/")
                rec_writers = None
                rec_start   = None
        elif key == ord('s'):
            fname = f"thermal_{dt.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fname, canvas)
            print(f"Saved: {fname}")
        elif key == ord('k') or key == ord('K'):
            global _cal_raw_hot
            # 99th percentile = hottest region; robust against single hot pixels
            _cal_raw_hot = int(np.percentile(u16_raw, 99))
            raw_t = _u16_to_raw_celsius(np.array(_cal_raw_hot, dtype=np.uint16)).item()
            print(f"  [K] hot reference: u16={_cal_raw_hot}  raw formula={raw_t:.1f}°C → target {_cal_temp_hot:.0f}°C")
            print(f"      frame u16: min={u16_raw.min()}  median={int(np.median(u16_raw))}  99pct={_cal_raw_hot}  max={u16_raw.max()}")
            _update_calibration()
            save_calibration()
        elif key == ord('a') or key == ord('A'):
            global _cal_raw_amb
            _cal_raw_amb = int(np.median(u16_raw))
            raw_t = _u16_to_raw_celsius(np.array(_cal_raw_amb, dtype=np.uint16)).item()
            print(f"  [A] ambient reference: u16={_cal_raw_amb}  raw formula={raw_t:.1f}°C → target {_cal_temp_amb:.0f}°C")
            print(f"      frame u16: min={u16_raw.min()}  median={_cal_raw_amb}  99pct={int(np.percentile(u16_raw,99))}  max={u16_raw.max()}")
            _update_calibration()
            save_calibration()
        elif key == ord('0'):
            _cal_raw_hot = None
            _cal_raw_amb = None
            _update_calibration()
            save_calibration()
            print("  Calibration reset and file cleared.")
        elif key in (ord('p'), ord('P'), ord('o'), ord('O')):
            path = _pick_replay_file()
            if path:
                cv2.destroyAllWindows()
                replay_main(path)
                # Re-open live window after replay closes
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
                cv2.resizeWindow(WIN, init_w, init_h)
        elif key == ord('d') or key == ord('D'):
            raw_T_valid = _u16_to_raw_celsius(u16_raw)
            raw_T_valid = raw_T_valid[(raw_T_valid > -40) & (raw_T_valid < 400)]
            print(f"  [D] u16: min={u16_raw.min()}  median={int(np.median(u16_raw))}  "
                  f"99pct={int(np.percentile(u16_raw,99))}  max={u16_raw.max()}")
            print(f"      raw formula T: min={raw_T_valid.min():.1f}  mean={raw_T_valid.mean():.1f}  "
                  f"max={raw_T_valid.max():.1f} °C")
            print(f"      calibration: slope={_cal_slope:.4f}  offset={_cal_offset:+.2f}°C  "
                  f"({'active' if _cal_raw_hot or _cal_raw_amb else 'none'})")

    if rec_writers is not None:
        rec_writers[0].release()
        rec_writers[1].release()
        print(f"  Recording finalised → {rec_folder}/")
    cam.release()
    cv2.destroyAllWindows()


def replay_main(video_path):
    """Replay a previously recorded raw video file through the neural net."""
    print(f"Replay mode: {video_path}")
    if not os.path.exists(video_path):
        print(f"ERROR: File not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_in    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Source: {w_in}x{h_in} @ {fps_in:.1f}fps, {total} frames")
    print("Keys: [q] quit  [f] fullscreen  [space] pause/play  [</>] step frame")
    print("      Use the slider to scrub through the video")

    WIN = "Thermal Replay"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

    tracker = HotBlobTracker()

    disp_w = THERMAL_W * SCALE
    disp_h = THERMAL_H * SCALE
    canvas_w = MARGIN_L + disp_w + MARGIN_R
    canvas_h = MARGIN_TOP + disp_h + MARGIN_BOT
    cv2.resizeWindow(WIN, canvas_w, canvas_h)

    # ── Slider state ──────────────────────────────────────────────────────
    slider_seeking  = [False]   # mutable so callback can set it
    slider_target   = [0]
    slider_suppress = [False]   # suppress callback during programmatic updates

    def _on_slider(pos):
        if slider_suppress[0]:
            return
        slider_seeking[0] = True
        slider_target[0]  = pos

    cv2.createTrackbar("Frame", WIN, 0, max(total - 1, 1), _on_slider)

    frame_num = 0
    paused = False
    delay  = max(1, int(1000 / fps_in))
    frame  = None

    # ── Replay output video ───────────────────────────────────────────────
    src_dir  = os.path.dirname(video_path)
    src_base = os.path.splitext(os.path.basename(video_path))[0]
    replay_out_path = os.path.join(src_dir, f"{src_base}_REPLAY.avi")
    replay_writer   = cv2.VideoWriter(
        replay_out_path, cv2.VideoWriter_fourcc(*'MJPG'),
        fps_in, (canvas_w, canvas_h))
    print(f"  Saving replay to: {replay_out_path}")

    while True:
        # ── Handle slider seek ────────────────────────────────────────────
        if slider_seeking[0]:
            target = slider_target[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if ret:
                frame_num = target + 1
                tracker = HotBlobTracker()   # reset tracker on seek
            slider_seeking[0] = False
            paused = True
        elif not paused:
            ret, frame = cap.read()
            if not ret:
                print("  End of video.")
                paused = True
                if frame is None:
                    break
                continue
            frame_num += 1
            # Update slider position (without triggering callback)
            slider_suppress[0] = True
            cv2.setTrackbarPos("Frame", WIN, frame_num - 1)
            slider_suppress[0] = False

        if frame is None:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num = 1

        sensor = cv2.resize(frame, (THERMAL_W, THERMAL_H),
                            interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(sensor, cv2.COLOR_BGR2GRAY)
        T = gray.astype(np.float32) * 0.5 + 10.0
        T[gray < 20] = np.nan

        now = time.time()

        mn, mx = int(gray.min()), int(gray.max())
        if mx > mn:
            norm = ((gray.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            norm = gray.copy()

        gray_bgr = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        colored  = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        T_valid_disp = T[~np.isnan(T)]
        if len(T_valid_disp):
            pct_thresh  = float(np.percentile(T_valid_disp, HOT_DISPLAY_PCT))
            stat_thresh = float(T_valid_disp.mean() + 3.0 * max(T_valid_disp.std(), 0.1))
            hot_thresh  = max(pct_thresh, stat_thresh)
        else:
            hot_thresh = 999.0
        hot_mask = ((~np.isnan(T)) & (T >= hot_thresh)).astype(np.uint8)
        combined = np.where(hot_mask[:, :, np.newaxis], colored, gray_bgr)
        thermal  = cv2.resize(combined, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        ox, oy = MARGIN_L, MARGIN_TOP
        canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)
        canvas[oy:oy+disp_h, ox:ox+disp_w] = thermal

        confirmed = tracker.update(gray, T)
        draw_secondary_blobs(canvas, tracker.last_blobs, confirmed, ox, oy)
        if confirmed:
            draw_hot_blob(canvas, confirmed, tracker, disp_w, disp_h, ox, oy, False)

        panel_x = ox + disp_w + 4
        draw_candidate_panel(canvas, tracker.sticky_blobs, confirmed,
                             thermal, panel_x, oy, MARGIN_R - 8, disp_h, now)

        img_cx, img_cy = ox + disp_w//2, oy + disp_h//2
        cv2.drawMarker(canvas, (img_cx, img_cy), (0,255,0),
                       cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)

        # HUD
        pct = frame_num * 100 // max(total, 1)
        t_sec = frame_num / fps_in
        t_total = total / fps_in
        progress = f"Frame {frame_num}/{total}  ({pct}%)  {t_sec:.1f}s / {t_total:.1f}s"
        status   = "PAUSED" if paused else "PLAYING"
        draw_text(canvas, f"REPLAY  {progress}  [{status}]",
                  (ox, MARGIN_TOP - 8), scale=0.5, shadow=False)
        draw_text(canvas, os.path.basename(video_path),
                  (ox, canvas_h - 10), scale=0.4, shadow=False, color=(140,140,140))

        # Playback controls hint
        ctrl = "Space=play/pause  <>=step  Slider=scrub  O=open  Q=quit"
        draw_text(canvas, ctrl, (ox, canvas_h - 28), scale=0.35,
                  shadow=False, color=(100, 100, 100))

        cv2.imshow(WIN, canvas)
        if not paused:
            replay_writer.write(canvas)

        key = cv2.waitKey(delay if not paused else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f') or key == ord('F'):
            fs = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fs != cv2.WINDOW_FULLSCREEN else cv2.WINDOW_NORMAL)
        elif key == ord(' '):
            paused = not paused
        elif key == ord('.') or key == ord('>'):
            paused = True
            ret, frame = cap.read()
            if not ret:
                print("  End of video.")
                continue
            frame_num += 1
            slider_suppress[0] = True
            cv2.setTrackbarPos("Frame", WIN, frame_num - 1)
            slider_suppress[0] = False
        elif key == ord(',') or key == ord('<'):
            paused = True
            pos = max(0, frame_num - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                frame_num = pos + 1
                slider_suppress[0] = True
            cv2.setTrackbarPos("Frame", WIN, frame_num - 1)
            slider_suppress[0] = False
        elif key in (ord('o'), ord('O')):
            new_path = _pick_replay_file()
            if new_path:
                # Close current replay and start new one
                replay_writer.release()
                print(f"  Replay saved: {replay_out_path}")
                cap.release()
                cv2.destroyAllWindows()
                replay_main(new_path)
                return

    replay_writer.release()
    print(f"  Replay saved: {replay_out_path}")
    cap.release()
    cv2.destroyAllWindows()


def _find_latest_recording():
    """Find the most recent raw recording file."""
    rec_base = os.path.join(os.getcwd(), "thermal_recordings")
    if not os.path.isdir(rec_base):
        return None
    runs = sorted([d for d in os.listdir(rec_base)
                   if os.path.isdir(os.path.join(rec_base, d))], reverse=True)
    for run in runs:
        run_dir = os.path.join(rec_base, run)
        for f in sorted(os.listdir(run_dir)):
            if f.startswith("raw") and f.endswith(".avi"):
                return os.path.join(run_dir, f)
    return None


_LAST_REPLAY_FILE = os.path.expanduser("~/.hikcam_last_replay")
_last_replay_path = None   # track last replayed file

def _load_last_replay():
    global _last_replay_path
    if os.path.exists(_LAST_REPLAY_FILE):
        try:
            p = open(_LAST_REPLAY_FILE).read().strip()
            if os.path.exists(p):
                _last_replay_path = p
        except Exception:
            pass

def _save_last_replay(path):
    global _last_replay_path
    _last_replay_path = path
    try:
        with open(_LAST_REPLAY_FILE, 'w') as f:
            f.write(path)
    except Exception:
        pass

_load_last_replay()

def _pick_replay_file(mode="browse"):
    """
    Open a dialog to pick a recording.
    mode: "browse" = file chooser, "last_recorded" = latest recording,
          "last_replayed" = re-open last replayed file
    """
    global _last_replay_path

    if mode == "last_replayed" and _last_replay_path and os.path.exists(_last_replay_path):
        return _last_replay_path
    if mode == "last_recorded":
        path = _find_latest_recording()
        if path:
            _save_last_replay(path)
        return path

    import tkinter as tk
    from tkinter import filedialog, ttk

    root = tk.Tk()
    root.title("Open Recording")
    root.geometry("400x160")
    root.resizable(False, False)

    result = [None]

    def do_browse():
        latest = _find_latest_recording()
        initial_dir = os.path.dirname(latest) if latest else os.getcwd()
        path = filedialog.askopenfilename(
            parent=root, title="Open recording for replay",
            initialdir=initial_dir,
            initialfile=os.path.basename(latest) if latest else "",
            filetypes=[("AVI/MP4 video", "*.avi *.mp4"), ("All files", "*.*")],
        )
        if path:
            result[0] = path
            root.destroy()

    def do_last_recorded():
        path = _find_latest_recording()
        if path:
            result[0] = path
            root.destroy()

    def do_last_replayed():
        if _last_replay_path and os.path.exists(_last_replay_path):
            result[0] = _last_replay_path
            root.destroy()

    def do_cancel():
        root.destroy()

    lbl = tk.Label(root, text="Open a recording for replay", font=("sans", 11))
    lbl.pack(pady=(12, 8))

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=4)

    tk.Button(btn_frame, text="Browse...", width=16, command=do_browse).grid(
        row=0, column=0, padx=4, pady=2)
    tk.Button(btn_frame, text="Last Recorded", width=16, command=do_last_recorded).grid(
        row=0, column=1, padx=4, pady=2)

    replay_btn = tk.Button(btn_frame, text="Last Replayed", width=16,
                           command=do_last_replayed)
    replay_btn.grid(row=1, column=0, padx=4, pady=2)
    if not (_last_replay_path and os.path.exists(_last_replay_path)):
        replay_btn.config(state=tk.DISABLED)

    tk.Button(btn_frame, text="Cancel", width=16, command=do_cancel).grid(
        row=1, column=1, padx=4, pady=2)

    root.mainloop()

    if result[0]:
        _save_last_replay(result[0])
    return result[0]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HIK Thermal Camera Viewer")
    parser.add_argument("--replay", type=str, nargs='?', const="__browse__",
                        default=None,
                        help="Replay a recorded file (opens browser if no path given)")
    args = parser.parse_args()
    if args.replay:
        if args.replay == "__browse__":
            path = _pick_replay_file()
            if not path:
                print("No file selected.")
                sys.exit(0)
        else:
            path = args.replay
        replay_main(path)
    else:
        main()
