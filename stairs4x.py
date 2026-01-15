# analyze_stairs_roundtrip_4phases_startfeet_headless.py
# Four-phase sequence: UP1 -> DOWN1 -> UP2 -> DOWN2
# Score = number of completed phases [0..4]
#
# Headless version:
#  - No display, no keyboard/mouse interaction, no main()
#  - Same automatic logic for:
#        * pose-based orientation
#        * TOP-line detection
#        * baseline START-line from feet
#        * 4-phase FSM (UP1, DOWN1, UP2, DOWN2)
#  - Returns a results dict

import cv2
import numpy as np
from collections import deque

try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("Please install: pip install mediapipe opencv-python numpy mediapipe") from e

# =========================
# User settings
# =========================
ANALYZE_MAX_FPS = 12
MIN_CONF = 0.5

# ---- Hough params (TOP edge; near-horizontal) ----
ANGLE_MAX_DEG_INIT   = 10
MIN_LINE_LENGTH_INIT = 80
MAX_LINE_GAP_INIT    = 10
HOUGH_THRESH_INIT    = 60
CANNY_LOW_INIT, CANNY_HIGH_INIT = 50, 150
Y_CLUSTER_BIN = 6
ROI_SHAVE_BOTTOM_TOP = 0.05   # ignore last bottom strip when detecting TOP edges

# ---- Relax schedule for TOP if strict fails ----
RELAX_SCHEDULE = [
    (12, 70, 12, 55, 40, 160, 0.03),
    (14, 60, 14, 50, 35, 170, 0.00),
    (16, 50, 16, 45, 30, 180, 0.00),
]

# ---- Smoothing for TOP line ----
EDGE_EMA_ALPHA = 0.30

# ---- Start detection ----
BASELINE_SEC = 1.0            # collect start-feet baseline for ~1s
START_MIN_FRAMES = 6
HIP_UP_VEL_PX_S   = 12.0      # |hip velocity| for movement onset
START_Y_DELTA_PX  = 30        # or feet-Y vs baseline
START_MANUAL_KEY  = False     # no manual key in headless version

# ---- Phase confirmations ----
CROSS_MARGIN_PX  = 15         # UP: feet must be ≥ this many px BELOW TOP line
RETURN_MARGIN_PX = 50         # DOWN: |feet_y - START_y| ≤ this
UP_DWELL_FRAMES   = 1         # set 2–3 if you want de-jitter
DOWN_DWELL_FRAMES = 2

# ---- Portrait/landscape heuristic (pose based) ----
VERTICAL_SPAN_THRESH = 0.1   # if head–feet span < 10% of height => rotate 90°


# =========================
# Helpers
# =========================
def create_pose():
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

def visible(lm, i):
    return lm and i < len(lm) and lm[i].visibility >= MIN_CONF

def get_xy(lm, i, w, h):
    return (lm[i].x * w, lm[i].y * h) if (lm and i < len(lm)) else (np.nan, np.nan)

def push(deq, v, maxlen):
    deq.append(np.nan if v is None else v)
    if len(deq) > maxlen:
        deq.popleft()

def median_last(vals, k):
    arr = [v for v in list(vals)[-k:] if v is not None and not np.isnan(v)]
    return float(np.median(arr)) if arr else np.nan

def should_rotate_by_pose(first_bgr, pose, span_thresh=VERTICAL_SPAN_THRESH):
    """
    Use MediaPipe pose on the first frame.
    If the vertical span (head->feet) is small relative to image height,
    assume the subject is sideways and rotate 90° clockwise.
    """
    h, w = first_bgr.shape[:2]
    rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_landmarks:
        print("[ORIENT] No pose; not rotating.")
        return False

    lm = res.pose_landmarks.landmark
    NOSE = 0
    L_ANK, R_ANK = 27, 28

    def safe_y(idx):
        return lm[idx].y * h if (idx < len(lm) and lm[idx].visibility >= MIN_CONF) else np.nan

    y_head = safe_y(NOSE)
    y_ankL = safe_y(L_ANK)
    y_ankR = safe_y(R_ANK)
    feet_candidates = [v for v in [y_ankL, y_ankR] if not np.isnan(v)]

    if np.isnan(y_head) or not feet_candidates:
        print("[ORIENT] Missing head/feet; not rotating.")
        return False

    y_feet = float(np.median(feet_candidates))
    dy = abs(y_feet - y_head) / float(h)

    print(f"[ORIENT] head–feet vertical span (fraction of height) = {dy:.3f}")
    if dy < span_thresh:
        print("[ORIENT] Span is small => rotating video 90° CW.")
        return True
    else:
        print("[ORIENT] Span is large => keeping as-is (no rotation).")
        return False


# =========================
# TOP edge detection (Hough)
# =========================
def detect_edges_hough_top(frame_bgr,
                           angle_max_deg=ANGLE_MAX_DEG_INIT,
                           min_len=MIN_LINE_LENGTH_INIT,
                           max_gap=MAX_LINE_GAP_INIT,
                           hough_thr=HOUGH_THRESH_INIT,
                           canny_low=CANNY_LOW_INIT,
                           canny_high=CANNY_HIGH_INIT,
                           shave_bottom=ROI_SHAVE_BOTTOM_TOP):
    """Return (edges_y_sorted, nearest_y, debug_img) for TOP edges using most of frame."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3, L2gradient=True)

    cutoff = int((1.0 - shave_bottom) * h)
    roi = edges[:cutoff, :]

    lines = cv2.HoughLinesP(
        roi, rho=1, theta=np.pi/180, threshold=hough_thr,
        minLineLength=min_len, maxLineGap=max_gap
    )

    y_list = []
    debug = img.copy()
    if lines is not None:
        max_slope = np.tan(np.deg2rad(angle_max_deg))
        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx, dy = float(x2 - x1), float(y2 - y1)
            if dx == 0:
                continue
            slope = dy / dx
            if abs(slope) <= max_slope and abs(x2 - x1) >= min_len:
                ymid = int((y1 + y2) // 2)
                y_list.append(ymid)

    if not y_list:
        return [], None, debug

    y_list.sort()
    merged, cur = [], [y_list[0]]
    for y in y_list[1:]:
        if abs(y - cur[-1]) <= Y_CLUSTER_BIN:
            cur.append(y)
        else:
            merged.append(int(np.median(cur)))
            cur = [y]
    merged.append(int(np.median(cur)))
    merged = sorted(set(merged))
    nearest_y = max(merged)  # nearest to camera (largest y)

    return merged, nearest_y, debug

def detect_top_first_robust(frame_bgr):
    ylist, near, dbg = detect_edges_hough_top(frame_bgr)
    if near is not None:
        return ylist, near, ("strict", dbg)
    for (ang, mlen, gap, hthr, cl, ch, shave) in RELAX_SCHEDULE:
        ylist, near, dbg = detect_edges_hough_top(
            frame_bgr, ang, mlen, gap, hthr, cl, ch, shave
        )
        if near is not None:
            return ylist, near, ("relaxed", dbg)
    return [], None, ("none", frame_bgr.copy())


# =========================
# Core analysis (headless)
# =========================
def analyze_roundtrip_4phases(video_path: str):
    """
    Analyze stairs roundtrip (UP1, DOWN1, UP2, DOWN2) in a headless way.

    Returns:
        dict with keys like:
        {
          "video_path": str,
          "rotated_cw": bool,
          "top_edges": [y1, y2, ...],
          "top_nearest_y": float or None,
          "start_line_y": float or None,
          "start_time": float or None,
          "phase_times": {
              "UP1_FIN": float or None,
              "DOWN1_FIN": float or None,
              "UP2_FIN": float or None,
              "DOWN2_FIN": float or None,
          },
          "phases_completed": int,
          "config": {...}
        }
    """
    pose = create_pose()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame.")

    # ---- Decide orientation from first frame using pose ----
    rotate_cw = should_rotate_by_pose(first, pose)
    if rotate_cw:
        first = cv2.rotate(first, cv2.ROTATE_90_CLOCKWISE)

    # Detect TOP from (possibly rotated) first frame
    edges_y, near_y, (tier, dbg_top) = detect_top_first_robust(first)
    print(f"[TOP] tier={tier}, count={len(edges_y)}, nearest_y={near_y}")

    v_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if v_fps < 1:
        v_fps = 30.0
    frame_interval = max(1, int(round(v_fps / ANALYZE_MAX_FPS)))
    eff_fps = (v_fps / frame_interval) if frame_interval > 0 else v_fps

    # Landmarks
    L_ANK, R_ANK = 27, 28
    L_HIP, R_HIP = 23, 24

    # Histories
    feet_y_hist = deque(maxlen=240)
    hip_y_hist  = deque(maxlen=240)

    # Baseline buffers
    baseline_frames = int(BASELINE_SEC * max(1.0, eff_fps))
    feet_y_samples = []
    baseline_locked = False
    y_start_fixed = np.nan

    # State & timestamps
    started = False
    start_time = None
    y_top_smooth = float(near_y) if near_y is not None else np.nan

    phases = ["UP1", "DOWN1", "UP2", "DOWN2"]
    phase_idx = 0  # current phase pointer
    phase_times = {p: None for p in ["UP1_FIN", "DOWN1_FIN", "UP2_FIN", "DOWN2_FIN"]}
    phase_dwell = 0

    first_ts = None
    frame_idx = 1

    # Predicates
    def armed_up(y_feet, y_top):
        return (y_feet - y_top) >= CROSS_MARGIN_PX

    def armed_down(y_feet, y_start):
        return abs(y_feet - y_start) <= RETURN_MARGIN_PX

    # Rewind to start of video (we already used first frame only for orientation/edges)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        # Apply same rotation decision to all frames
        if rotate_cw:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        t_abs = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if first_ts is None:
            first_ts = t_abs
        t_rel = t_abs - first_ts
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        lm = res.pose_landmarks.landmark if res.pose_landmarks else None

        # ---- Feet line
        ankL = ankR = (np.nan, np.nan)
        hip_mid = (np.nan, np.nan)
        if lm:
            if visible(lm, L_ANK):
                ankL = get_xy(lm, L_ANK, w, h)
            if visible(lm, R_ANK):
                ankR = get_xy(lm, R_ANK, w, h)
            if visible(lm, L_HIP) and visible(lm, R_HIP):
                lhip = get_xy(lm, L_HIP, w, h)
                rhip = get_xy(lm, R_HIP, w, h)
                hip_mid = ((lhip[0] + rhip[0]) * 0.5, (lhip[1] + rhip[1]) * 0.5)

        feet_candidates = [v for v in [ankL[1], ankR[1]] if not np.isnan(v)]
        y_feet = float(np.median(feet_candidates)) if feet_candidates else np.nan

        # Smooth
        push(feet_y_hist, y_feet, feet_y_hist.maxlen)
        push(hip_y_hist, hip_mid[1], hip_y_hist.maxlen)
        y_feet_med = median_last(feet_y_hist, 5)

        # --- Build start-feet baseline (only before movement starts)
        if (not baseline_locked and
                len(feet_y_samples) < baseline_frames and
                not np.isnan(y_feet_med)):
            feet_y_samples.append(y_feet_med)

        if (not baseline_locked) and len(feet_y_samples) >= max(3, baseline_frames // 2):
            y_start_fixed = float(np.median(feet_y_samples))
            baseline_locked = True
            print(f"[START LINE] locked from baseline: y={int(round(y_start_fixed))}")

        # Start detection (movement onset)
        if (not started and frame_idx >= START_MIN_FRAMES and
                len(feet_y_samples) >= 3):
            ys = [y for y in list(hip_y_hist)[-6:] if not np.isnan(y)]
            hip_vel = 0.0
            if len(ys) >= 2:
                dt = (len(ys) - 1) / max(1.0, eff_fps)
                hip_vel = abs((ys[-2] - ys[-1]) / dt) if dt > 0 else 0.0
            feet_y_base = float(np.median(feet_y_samples)) if feet_y_samples else np.nan
            feet_delta_ok = (not np.isnan(feet_y_base) and
                             not np.isnan(y_feet_med) and
                             (y_feet_med - feet_y_base) >= START_Y_DELTA_PX)
            if (hip_vel > HIP_UP_VEL_PX_S) or feet_delta_ok:
                started = True
                start_time = t_rel
                print(f"[START] t={t_rel:.2f}s (|hip_vel|={hip_vel:.1f} px/s, "
                      f"feet_delta={'OK' if feet_delta_ok else 'no'})")

        # Keep TOP stable vs first detection
        if near_y is not None and not np.isnan(y_top_smooth):
            y_top_smooth = EDGE_EMA_ALPHA * near_y + (1 - EDGE_EMA_ALPHA) * y_top_smooth

        # ------------------------
        # Phase machine
        # ------------------------
        current_phase = phases[phase_idx] if phase_idx < len(phases) else None

        if started and current_phase and not np.isnan(y_feet_med):
            if current_phase.startswith("UP") and not np.isnan(y_top_smooth):
                armed = armed_up(y_feet_med, y_top_smooth)
                dwell_need = UP_DWELL_FRAMES
            elif current_phase.startswith("DOWN") and not np.isnan(y_start_fixed):
                armed = armed_down(y_feet_med, y_start_fixed)
                dwell_need = DOWN_DWELL_FRAMES
            else:
                armed = False
                dwell_need = 1

            if armed:
                phase_dwell += 1
                if phase_dwell >= dwell_need:
                    # Phase completed
                    tag = current_phase + "_FIN"
                    phase_times[tag] = t_rel
                    print(f"[{current_phase} FINISH] t={t_rel:.2f}s")
                    phase_idx += 1
                    phase_dwell = 0
                    if phase_idx >= len(phases):
                        # All 4 phases done
                        break
            else:
                phase_dwell = max(0, phase_dwell - 1)

        # stop if all phases are done
        if phase_idx >= len(phases):
            break

    cap.release()
    pose.close()

    # =========================
    # Results & PHASE score
    # =========================
    completed = sum(
        1 for tag in ["UP1_FIN", "DOWN1_FIN", "UP2_FIN", "DOWN2_FIN"]
        if phase_times[tag] is not None
    )

    print("\n=== RESULTS (4-Phase Round Trip) ===")
    print(f"Score (phases completed): {completed} / 4")
    for tag in ["UP1_FIN", "DOWN1_FIN", "UP2_FIN", "DOWN2_FIN"]:
        t = phase_times[tag]
        print(f"{tag}: {'—' if t is None else f'{t:.2f}s'}")

    results = {
        "top_edges": edges_y,
        "top_nearest_y": float(near_y) if near_y is not None else None,
        "start_line_y": float(y_start_fixed) if not np.isnan(y_start_fixed) else None,
        "start_time": float(start_time) if start_time is not None else None,
        "phase_times": {
            k: (float(v) if v is not None else None)
            for k, v in phase_times.items()
        },
        "score": int(completed),
    }
    return results
