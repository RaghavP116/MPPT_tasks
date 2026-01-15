"""
Chair Rise (5x) – Record & Analyze (FSM-based, headless)
--------------------------------------------------------
- Records 20s from default webcam to chair_rise.mp4 (same folder).
- Analyzes video with MediaPipe Pose:
    * Counts 5 sit↔stand cycles using a simple state machine:
        CALIBRATING -> SITTING -> RISING -> STANDING -> SITTING_DOWN -> SITTING ...
    * Timer:
        - Starts at first RISING from SITTING (first time person starts to get up)
        - Ends when 5th cycle returns to SITTING (5 sit→stand→sit completed)
        - Or when video ends
    * Arms:
        - Checks if arms are folded across chest while doing the task.
        - Prints time segments where arms are NOT folded.
    * Movement:
        - Tracks hip & shoulder "rise" vs sitting baseline
          (normalized by body height).
        - Prints summary of hip & shoulder momentum over time.
"""

import cv2
import time
import math
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("Please install mediapipe: pip install mediapipe") from e

# ----------------- Config -----------------
RECORD_DURATION_SEC = 20
RECORD_FPS = 30
ANALYZE_MAX_FPS = 12          # process at <= this FPS
CALIBRATION_SEC = 2.0         # first 2s are sitting baseline
MIN_CONF = 0.5                # minimum landmark visibility

# Normalized hip-rise thresholds (fractions of body height)
# y axis increases downward; "rise" = baseline_hip_y - hip_y (normalized)
RISE_START_NORM = 0.10        # start rising from sitting
RISE_PEAK_NORM  = 0.26        # considered "standing" above this
FALL_START_NORM = 0.18        # start going down from standing
SIT_DONE_NORM   = 0.07        # back to sitting (near baseline)

# For reporting "ideal" ranges (purely informational)
IDEAL_HIP_RISE_FRAC      = RISE_PEAK_NORM
IDEAL_SHOULDER_RISE_FRAC = 0.10

# Arms-folded tolerance (relative to shoulder width)
WRIST_CHEST_H_FRAC = 0.75
WRIST_NEAR_SHOULDER_FRAC = 1.05
ELBOW_FLEX_MAX_DEG = 120.0
HAND_VIOLATION_MIN_SEC = 0.20

# Scoring bands based on total time (first rise → 5th completion)
def score_from_time(total_time):
    if total_time < 11.0:
        return 4, "< 11.0 s"
    elif 11.1 <= total_time <= 13.9:
        return 3, "11.1–13.9 s"
    elif 14.0 <= total_time <= 16.9:
        return 2, "14.0–16.9 s"
    else:
        return 1, ">= 17.0 s"

# --------------------------------------------
# Utility helpers
# --------------------------------------------
def angle(a, b, c):
    """Angle ABC (in degrees) given 2D points a, b, c."""
    bax = a[0] - b[0]; bay = a[1] - b[1]
    bcx = c[0] - b[0]; bcy = c[1] - b[1]
    dot = bax * bcx + bay * bcy
    na = math.hypot(bax, bay); nc = math.hypot(bcx, bcy)
    if na == 0 or nc == 0:
        return 180.0
    cosang = max(-1.0, min(1.0, dot / (na * nc)))
    return math.degrees(math.acos(cosang))

def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

def median_or_none(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return float(np.median(vals)) if vals else None

def visible(lm, idx):
    return (lm[idx].visibility if lm and 0 <= idx < len(lm) else 0.0) >= MIN_CONF

def get_xy(lm, idx, w, h):
    if not lm or idx >= len(lm):
        return None
    return (lm[idx].x * w, lm[idx].y * h)

def within(v, lo, hi):
    return v is not None and lo <= v <= hi

# --------------------------------------------
# Arms folded heuristic
# --------------------------------------------
def arms_folded_ok(lm, w, h):
    """
    Arms folded across chest:
    - Wrists in front of torso (between shoulder x-range)
    - Vertically between shoulders and hips
    - Near chest center horizontally
    - Elbows bent (angle <= ELBOW_FLEX_MAX_DEG)
    - Wrists roughly near opposite shoulders (crossed)
    """
    L_SH, R_SH = 11, 12
    L_EL, R_EL = 13, 14
    L_WR, R_WR = 15, 16
    L_HIP, R_HIP = 23, 24

    needed = [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP]
    if any(not visible(lm, i) for i in needed):
        return None

    lsh = get_xy(lm, L_SH, w, h); rsh = get_xy(lm, R_SH, w, h)
    lel = get_xy(lm, L_EL, w, h); rel = get_xy(lm, R_EL, w, h)
    lwr = get_xy(lm, L_WR, w, h); rwr = get_xy(lm, R_WR, w, h)
    lhp = get_xy(lm, L_HIP, w, h); rhp = get_xy(lm, R_HIP, w, h)

    sh_y  = (lsh[1] + rsh[1]) * 0.5
    hip_y = (lhp[1] + rhp[1]) * 0.5

    chest_x = (lsh[0] + rsh[0]) * 0.5
    shoulder_w = abs(rsh[0] - lsh[0]) + 1e-6

    # vertical band (shoulder→hip)
    y_ok_left  = within(lwr[1], min(sh_y, hip_y), max(sh_y, hip_y))
    y_ok_right = within(rwr[1], min(sh_y, hip_y), max(sh_y, hip_y))

    # near chest center horizontally
    x_ok_left  = abs(lwr[0] - chest_x) <= WRIST_CHEST_H_FRAC * shoulder_w
    x_ok_right = abs(rwr[0] - chest_x) <= WRIST_CHEST_H_FRAC * shoulder_w

    # elbows bent
    ang_left  = angle(lsh, lel, lwr)
    ang_right = angle(rsh, rel, rwr)
    elbow_ok_left  = ang_left  <= ELBOW_FLEX_MAX_DEG
    elbow_ok_right = ang_right <= ELBOW_FLEX_MAX_DEG

    # wrists roughly "crossed" near opposite shoulders
    cross_left  = dist(lwr, rsh) <= WRIST_NEAR_SHOULDER_FRAC * shoulder_w
    cross_right = dist(rwr, lsh) <= WRIST_NEAR_SHOULDER_FRAC * shoulder_w

    ok = (y_ok_left and y_ok_right and x_ok_left and x_ok_right
          and elbow_ok_left and elbow_ok_right
          and cross_left and cross_right)
    return bool(ok)

# --------------------------------------------
# Analysis (FSM-based sit/stand detector, returns results)
# --------------------------------------------
def analyze_video(filename):
    mp_pose_mod = mp.solutions.pose
    pose = mp_pose_mod.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {filename}")

    v_fps = cap.get(cv2.CAP_PROP_FPS)
    if v_fps <= 1:
        v_fps = RECORD_FPS
    frame_interval = max(1, int(round(v_fps / ANALYZE_MAX_FPS)))

    # landmark indices
    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24
    L_ANK, R_ANK = 27, 28

    # Calibration buffers
    calib_hip_y, calib_sh_y, calib_span = [], [], []
    baseline_hip_y = baseline_sh_y = baseline_span = None
    baseline_locked = False

    # FSM
    state = "CALIBRATING"  # CALIBRATING, SITTING, RISING, STANDING, SITTING_DOWN
    reps = 0

    # Timer
    first_ts = None
    timer_start = None
    timer_end = None
    manual_stop_flag = False  # reserved, but not used in headless mode

    # Arms violations
    hand_bad_segments = []
    hand_bad_active = False
    hand_bad_start = None

    # Movement tracking
    time_series = []
    hip_norm_series = []
    shoulder_norm_series = []

    t_rel = 0.0  # last relative time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_idx % frame_interval != 0:
            continue

        h, w = frame.shape[:2]
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if first_ts is None:
            first_ts = ts
        t_rel = ts - first_ts

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        lm = res.pose_landmarks.landmark if res.pose_landmarks else None

        hip_y = sh_y = span_y = None
        arms_ok = None

        need_core = [L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK]
        if lm and all(visible(lm, i) for i in need_core):
            lsh = get_xy(lm, L_SH, w, h); rsh = get_xy(lm, R_SH, w, h)
            lhp = get_xy(lm, L_HIP, w, h); rhp = get_xy(lm, R_HIP, w, h)
            lan = get_xy(lm, L_ANK, w, h); ran = get_xy(lm, R_ANK, w, h)

            sh_y  = (lsh[1] + rsh[1]) * 0.5
            hip_y = (lhp[1] + rhp[1]) * 0.5
            ank_y = (lan[1] + ran[1]) * 0.5
            span_y = abs(ank_y - sh_y)

            # Calibration phase
            if t_rel <= CALIBRATION_SEC:
                calib_hip_y.append(hip_y)
                calib_sh_y.append(sh_y)
                calib_span.append(span_y)

            # Lock baseline once after calibration
            if (not baseline_locked and t_rel > CALIBRATION_SEC
                    and len(calib_hip_y) >= 5):
                baseline_hip_y = median_or_none(calib_hip_y)
                baseline_sh_y  = median_or_none(calib_sh_y)
                baseline_span  = median_or_none(calib_span)
                baseline_locked = True
                state = "SITTING"
                print(
                    f"Baseline locked at t={t_rel:.2f}s | "
                    f"hip={baseline_hip_y:.1f}, span={baseline_span:.1f}, "
                    f"sh={baseline_sh_y:.1f}"
                )

        # Arms folded
        arms_ok = arms_folded_ok(lm, w, h) if lm else None
        if arms_ok is not None:
            if not arms_ok and not hand_bad_active:
                hand_bad_active = True
                hand_bad_start = t_rel
            elif arms_ok and hand_bad_active:
                if (t_rel - hand_bad_start) >= HAND_VIOLATION_MIN_SEC:
                    hand_bad_segments.append(
                        (round(hand_bad_start, 2), round(t_rel, 2))
                    )
                hand_bad_active = False
                hand_bad_start = None

        hip_norm = sh_norm = None
        if baseline_locked and hip_y is not None and baseline_span and baseline_span > 0:
            # Normalized rises (sitting baseline -> upward)
            hip_rise = baseline_hip_y - hip_y
            sh_rise  = baseline_sh_y - sh_y if baseline_sh_y is not None else 0.0
            hip_norm = hip_rise / baseline_span
            sh_norm  = sh_rise / baseline_span

            time_series.append(t_rel)
            hip_norm_series.append(hip_norm)
            shoulder_norm_series.append(sh_norm)

            # ------------- FSM for sit/stand ----------------
            # States: CALIBRATING, SITTING, RISING, STANDING, SITTING_DOWN

            if state == "SITTING":
                # start to rise
                if hip_norm is not None and hip_norm > RISE_START_NORM:
                    state = "RISING"
                    if timer_start is None:
                        timer_start = t_rel
                        print(f"Timer started at t={t_rel:.2f}s (first rise).")

            elif state == "RISING":
                # reached standing
                if hip_norm is not None and hip_norm > RISE_PEAK_NORM:
                    state = "STANDING"

            elif state == "STANDING":
                # start going down
                if hip_norm is not None and hip_norm < FALL_START_NORM:
                    state = "SITTING_DOWN"

            elif state == "SITTING_DOWN":
                # back to sitting
                if hip_norm is not None and hip_norm < SIT_DONE_NORM:
                    reps += 1
                    print(f"Rep {reps} completed at t={t_rel:.2f}s.")
                    state = "SITTING"
                    if reps >= 5 and timer_end is None and timer_start is not None:
                        timer_end = t_rel
                        print(
                            f"Timer ended at t={t_rel:.2f}s "
                            f"(5 reps completed)."
                        )

        # stop if 5 reps done and timer_end set
        if timer_end is not None and reps >= 5:
            break

    # Close any active arms-violation segment
    if hand_bad_active and hand_bad_start is not None:
        hand_bad_segments.append((round(hand_bad_start, 2), round(t_rel, 2)))

    cap.release()
    pose.close()

    # --------- Results & scoring ----------

    total_time = None
    score = None
    band = None
    status = "ok"

    if timer_start is not None and timer_end is not None:
        total_time = timer_end - timer_start
        print("\n=== CHAIR RISE (5x) – RESULTS ===")
        print(f"Timer start: {timer_start:.2f} s (first rise)")
        print(f"Timer end  : {timer_end:.2f} s")
        print(f"Total time : {total_time:.2f} s")

        if reps >= 5:
            score, band = score_from_time(total_time)
            print(f"Reps completed: {reps}/5")
            print(f"Score obtained: {score} points (band: {band})")
        else:
            status = "incomplete_reps"
            print(f"Reps completed: {reps}/5 (not all reps completed)")
            print("Score obtained: unable")
    else:
        status = "timer_failed"
        print("\n=== CHAIR RISE (5x) – RESULTS ===")
        print("Timer did not run correctly.")
        print(f"Reps completed: {reps}/5")
        print("Score obtained: unable")

    if manual_stop_flag:
        print("Note: Analysis stopped manually (flag reserved, no keybinding).")

    if hand_bad_segments:
        print("\nTimes when arms were NOT folded:")
        for a, b in hand_bad_segments:
            print(f"  - from {a:.2f}s to {b:.2f}s")
    else:
        print("\nNo arms-position violations detected.")

    # --------- Momentum summary ----------
    avg_hip = None
    avg_sh  = None
    if time_series and hip_norm_series and shoulder_norm_series:
        avg_hip = float(np.mean(hip_norm_series))
        avg_sh  = float(np.mean(shoulder_norm_series))
    else:
        print("\nInsufficient pose data to compute momentum profile.")

    print("=================================\n")

    # --------- RETURN structured results ----------
    result = {
        "reps_completed": reps,
        "timer_start_s": timer_start,
        "timer_end_s": timer_end,
        "total_time_s": total_time,
        "score_points": score,
        "score_band": band
    }
    return result

# --------------------------------------------
# Convenience wrapper: record + analyze
# --------------------------------------------
def run_chair_rise_assessment(path):
    """
    Records a new video and runs analysis, returning the same result dict
    that analyze_video() returns.
    """
    return analyze_video(path)