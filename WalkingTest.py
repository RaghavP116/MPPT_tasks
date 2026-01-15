
import cv2
import numpy as np
import time
from collections import deque

# =========================
# User Settings
# =========================
SEGMENT_FEET = 12.5         # per-leg distance
TARGET_SEGMENTS = 4         # four legs -> 50 ft total

# Geometry
CAMERA_HEIGHT_FEET = 3.0    # informational; not used when camera is level
KNOWN_DISTANCE_FEET = 8.0   # subject depth from camera (assumed constant)

# Camera intrinsics choice
USE_FOVX = True             # if True, compute fx from FOV_X_DEG
FOV_X_DEG = 70.0            # set your camera's true horizontal FOV if known
FX_PIX = None               # set if USE_FOVX=False, e.g., 950.0

# Robustness controls
JITTER_FLOOR_FT = 0.03      # ignore per-frame delta smaller than this
MAX_SPEED_FT_S = 12.0       # clamp extreme spikes (> typical sprint)
MEDIAN_WINDOW   = 5         # median filter window for per-frame deltas
VEL_GATE_FT_S   = 0.08      # treat as stationary if |v| < this
USE_FLOW        = True      # enable Lucas–Kanade optical flow refinement
FLOW_BLEND      = 0.35      # 0..1 trust on optical flow vs landmarks
VAR_MEAS        = 36.0      # Kalman measurement variance (↑ to trust landmarks less)
VAR_PROC_X      = 8.0       # Kalman process noise on position (↓ = smoother)
VAR_PROC_V      = 4.0       # Kalman process noise on velocity (↓ = smoother)

# Auto-stop range around 50 ft (45–55 ft)
AUTO_TARGET_FT       = SEGMENT_FEET * TARGET_SEGMENTS  # 50 ft
AUTO_TOLERANCE_FT    = 5.0
AUTO_MIN_FT          = AUTO_TARGET_FT - AUTO_TOLERANCE_FT  # 45
AUTO_MAX_FT          = AUTO_TARGET_FT + AUTO_TOLERANCE_FT  # 55

# =========================
# MediaPipe Pose
# =========================
try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("Please install mediapipe: pip install mediapipe") from e
mp_pose = mp.solutions.pose

# =========================
# Scale computation
# =========================
def compute_fx_from_fovx(frame_width_px: int, fovx_deg: float) -> float:
    fovx_rad = np.deg2rad(fovx_deg)
    return frame_width_px / (2.0 * np.tan(fovx_rad / 2.0))

def compute_pixels_per_foot(frame_width_px: int) -> float:
    if USE_FOVX:
        fx = compute_fx_from_fovx(frame_width_px, FOV_X_DEG)
    else:
        if FX_PIX is None:
            raise ValueError("Set FX_PIX (in pixels) when USE_FOVX=False")
        fx = float(FX_PIX)
    return fx / KNOWN_DISTANCE_FEET

# =========================
# Constant-Velocity Kalman Filter (fixed)
# State: [x, vx]^T ; measurement z = x
# =========================
class Kalman1D:
    """
    Constant-velocity Kalman filter for 1D position.
    State: [x, vx]^T, Measurement: z = x
    """
    def __init__(self, x0=0.0, vx0=0.0, var_process_x=5.0, var_process_v=2.0, var_meas=25.0):
        self.x = np.array([[x0], [vx0]], dtype=np.float32)   # [x; vx]
        self.P = np.eye(2, dtype=np.float32) * 1000.0        # big initial uncertainty
        self.qx = float(var_process_x)
        self.qv = float(var_process_v)
        self.R  = np.array([[float(var_meas)]], dtype=np.float32)

    def predict(self, dt: float):
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float32)
        # Simple, stable discrete process noise
        Qd = np.array([[self.qx * max(dt, 1e-3), 0.0],
                       [0.0,       self.qv * max(dt, 1e-3)]], dtype=np.float32)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Qd

    def update(self, z: float):
        H = np.array([[1.0, 0.0]], dtype=np.float32)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = np.array([[float(z)]], dtype=np.float32) - (H @ self.x)
        self.x = self.x + K @ y
        I = np.eye(2, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

    @property
    def pos(self): return float(self.x[0,0])
    @property
    def vel(self): return float(self.x[1,0])

# =========================
# Helpers (no drawing)
# =========================
def pelvis_proxy_xy(lm, W, H):
    # mp indices: 11(Lshoulder),12(Rshoulder),23(Lhip),24(Rhip)
    pts = []
    for idx in (11,12,23,24):
        p = lm[idx]
        if 0 <= p.x <= 1 and 0 <= p.y <= 1:
            pts.append((p.x*W, p.y*H))
    if len(pts) < 2:
        return None
    arr = np.array(pts, dtype=np.float32)
    cx, cy = np.mean(arr, axis=0)
    return np.array([cx, cy], dtype=np.float32)

def lk_refine(prev_gray, gray, prev_pt):
    # Lucas–Kanade optical flow for a single point
    p0 = np.array([[prev_pt]], dtype=np.float32)  # shape (1,1,2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None,
                                           winSize=(31,31), maxLevel=3,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
    if st is None or st[0,0] == 0:
        return None
    return p1[0,0]  # (x, y)

def run_stable_segment_counter(video_source):
    """
    Run the 12.5-ft segment counter headlessly on a webcam (0) or a video file.

    Returns:
        dict with keys:
            - total_distance_ft
            - target_distance_ft
            - auto_stop_range_ft (tuple)
            - auto_stop_triggered (bool)
            - manual_stop_triggered (bool)
            - stop_reason (str or None)
            - segments_done
            - target_segments
            - timer_started
            - timer_stopped
            - elapsed_time_s
            - average_speed_ft_s
            - px_per_foot
            - score_0_4             (50-ft walk test score)
            - score_label           (e.g. '<=15 s', '15.1–20 s', etc.)
            - meets_1mps            (True if ≥3.33 ft/s ≈ 1 m/s)
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        return {
            "error": "Could not open video source.",
        }

    ok, frame = cap.read()
    if not ok:
        cap.release()
        return {
            "error": "No frames from video source.",
        }

    H, W = frame.shape[:2]
    px_per_foot = compute_pixels_per_foot(W)

    thresholds = [SEGMENT_FEET * (i+1) for i in range(TARGET_SEGMENTS)]
    cum_feet = 0.0
    segments_done = 0

    # Filters
    kf = Kalman1D(var_process_x=VAR_PROC_X, var_process_v=VAR_PROC_V, var_meas=VAR_MEAS)
    prev_s = None
    delta_buf = deque(maxlen=MEDIAN_WINDOW)

    # Timing
    t_prev = time.time()
    start_time = None
    end_time   = None
    timer_started = False
    timer_stopped = False

    # Stop info
    stop_reason = None
    auto_stop_flag = False   # True when we hit 45–55 ft range
    manual_stop_flag = False # unused here (no key handling)

    # Flow buffers
    prev_gray = None
    prev_kalman_pos_px = None

    with mp_pose.Pose(model_complexity=1,
                      enable_segmentation=False,
                      smooth_landmarks=True,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                stop_reason = stop_reason or "video/stream ended"
                break

            t_now = time.time()
            dt = max(1e-3, t_now - t_prev)   # seconds
            t_prev = t_now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            meas_x = None
            pelvis_xy = None
            vx_ft_s = 0.0  # not returned per-frame, but used for gating

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                pelvis_xy = pelvis_proxy_xy(lm, W, H)
                if pelvis_xy is not None:
                    cx, cy = float(pelvis_xy[0]), float(pelvis_xy[1])

                    # Optical-flow refinement
                    if USE_FLOW:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if prev_gray is not None and prev_kalman_pos_px is not None:
                            flow_pt = lk_refine(
                                prev_gray,
                                gray,
                                np.array([prev_kalman_pos_px, cy], dtype=np.float32)
                            )
                            if flow_pt is not None:
                                fx = float(flow_pt[0])
                                cx = (1.0 - FLOW_BLEND) * cx + FLOW_BLEND * fx
                        prev_gray = gray

                    meas_x = cx

            # Kalman predict/update when we have a measurement
            if meas_x is not None:
                kf.predict(dt)
                kf.update(meas_x)
                kx = kf.pos
                kv = kf.vel
                prev_kalman_pos_px = kx

                # delta in pixels from filtered position
                if prev_s is not None:
                    delta_px = abs(kx - prev_s)
                    delta_ft_raw = delta_px / px_per_foot

                    # Median-of-deltas
                    delta_buf.append(delta_ft_raw)
                    delta_ft = float(np.median(delta_buf))

                    # Velocity gate (from Kalman) + clamp
                    vx_ft_s = kv / px_per_foot
                    if abs(vx_ft_s) < VEL_GATE_FT_S:
                        delta_ft = 0.0
                    max_delta = MAX_SPEED_FT_S * dt
                    if delta_ft > max_delta:
                        delta_ft = max_delta

                    # Jitter floor
                    if delta_ft < JITTER_FLOOR_FT:
                        delta_ft = 0.0

                    # -------- TIMER START: first non-zero increment --------
                    if delta_ft > 0.0 and not timer_started:
                        timer_started = True
                        start_time = t_now

                    # Update cumulative distance
                    cum_feet += delta_ft

                    # Segment thresholds
                    while (
                        segments_done < TARGET_SEGMENTS
                        and cum_feet >= thresholds[segments_done] - 1e-6
                    ):
                        segments_done += 1

                    # -------- AUTO STOP: 45–55 ft window --------
                    if (
                        timer_started and not timer_stopped and
                        AUTO_MIN_FT <= cum_feet <= AUTO_MAX_FT
                    ):
                        timer_stopped = True
                        end_time = t_now
                        auto_stop_flag = True
                        stop_reason = stop_reason or (
                            f"auto: distance {cum_feet:.2f} ft in "
                            f"[{AUTO_MIN_FT}, {AUTO_MAX_FT}]"
                        )
                        break

                prev_s = kx

    cap.release()
    cv2.destroyAllWindows()

    # =========================
    # Build timing + speed
    # =========================
    elapsed = None
    avg_speed = None

    if not timer_started:
        elapsed = None
        avg_speed = None
    elif timer_started and end_time is None:
        elapsed = None
        avg_speed = None
    else:
        elapsed = end_time - start_time
        if elapsed > 0:
            avg_speed = cum_feet / elapsed
        else:
            avg_speed = None

    # =========================
    # 50-ft walk test score (0–4)
    # =========================
    score = None
    score_label = None
    meets_1mps = None
    # 50 ft in 15 s → 3.33 ft/s ≈ 1 m/s
    one_mps_ft_per_s = AUTO_TARGET_FT / 15.0  # 50/15 ≈ 3.33

    if elapsed is None:
        # No valid time → "unable"
        score = 0
        score_label = "unable"
        meets_1mps = False
    else:
        if elapsed <= 15.0:
            score = 4
            score_label = "<=15 s"
        elif elapsed <= 20.0:
            score = 3
            score_label = "15.1–20 s"
        elif elapsed <= 25.0:
            score = 2
            score_label = "20.1–25 s"
        else:
            score = 1
            score_label = ">25 s"

        if avg_speed is not None:
            meets_1mps = avg_speed >= one_mps_ft_per_s
        else:
            meets_1mps = False

    results = {
        "total_distance_ft": float(cum_feet),
        "target_distance_ft": float(AUTO_TARGET_FT),
        "stop_reason": stop_reason,
        "timer_started": bool(timer_started),
        "timer_stopped": bool(timer_stopped),
        "elapsed_time_s": float(elapsed) if elapsed is not None else None,

        # 50-ft walk test scoring
        "score": int(score) if score is not None else None,
        "score_label": score_label,
    }

    return results