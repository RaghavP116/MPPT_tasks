# analyze_stairs_feet_gate_instant.py
# FINISH as soon as the FEET LINE (median Y of L/R ankles) is BELOW the nearest edge by >= margin.
# No velocity check. Optional tiny dwell. Now with scoring bands printed at the end.
#
# Keys: q=quit, f=fullscreen, e=redetect edge on current frame, m=click edge line, s=force start
# Requirements: pip install mediapipe opencv-python numpy

import os, cv2, numpy as np
from collections import deque

try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("Please install: pip install mediapipe opencv-python numpy mediapipe") from e

# =========================
# User settings
# =========================
VIDEO_FILENAME = "IMG_1680.MOV"
ANALYZE_MAX_FPS = 12
MIN_CONF = 0.5

# ---- Edge detection (Hough) ----
ANGLE_MAX_DEG_INIT   = 10
MIN_LINE_LENGTH_INIT = 80
MAX_LINE_GAP_INIT    = 10
HOUGH_THRESH_INIT    = 60
CANNY_LOW_INIT, CANNY_HIGH_INIT = 50, 150
Y_CLUSTER_BIN = 6
ROI_SHAVE_BOTTOM = 0.05  # ignore last bottom strip (foot clutter)

# If strict fails, relax gradually
RELAX_SCHEDULE = [
    (12, 70, 12, 55, 40, 160, 0.03),
    (14, 60, 14, 50, 35, 170, 0.00),
    (16, 50, 16, 45, 30, 180, 0.00),
]

# ---- Edge smoothing ----
EDGE_EMA_ALPHA = 0.30

# ---- Start detection ----
BASELINE_SEC = 1.0
START_MIN_FRAMES = 6
HIP_UP_VEL_PX_S   = 12.0      # |hip velocity| threshold to start
START_Y_DELTA_PX  = 30        # or feet-Y delta vs baseline
START_MANUAL_KEY  = True

# ---- FINISH (feet line below edge) ----
CROSS_MARGIN_PX = 15          # blue must be at least 15px BELOW green
DWELL_FRAMES    = 1           # =1 => instant; increase to 2-3 if you want de-jitter

# ---- Display ----
WINDOW_NAME     = "Stairs — Feet-line gate (instant finish)"
FIT_TO_SCREEN   = True
START_FULLSCREEN= False
SAVE_DEBUG      = True

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

def visible(lm, i): return lm and i < len(lm) and lm[i].visibility >= MIN_CONF
def get_xy(lm, i, w, h): return (lm[i].x * w, lm[i].y * h) if (lm and i < len(lm)) else (np.nan, np.nan)

def push(deq, v, maxlen):
    deq.append(np.nan if v is None else v)
    if len(deq) > maxlen: deq.popleft()

def median_last(vals, k):
    arr = [v for v in list(vals)[-k:] if v is not None and not np.isnan(v)]
    return float(np.median(arr)) if arr else np.nan

def get_screen_size():
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        try:
            import tkinter as tk
            root = tk.Tk(); root.withdraw()
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            return int(w), int(h)
        except Exception:
            return 1920, 1080

SCREEN_W, SCREEN_H = get_screen_size()

def resize_keep_aspect(img, max_w, max_h):
    h, w = img.shape[:2]
    s = min(max_w / float(w), max_h / float(h))
    return cv2.resize(img, (max(1,int(w*s)), max(1,int(h*s))), interpolation=cv2.INTER_AREA)

def init_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, SCREEN_W, SCREEN_H)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    if START_FULLSCREEN:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def toggle_fullscreen():
    cur = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    new_state = cv2.WINDOW_NORMAL if int(cur) == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, new_state)

# =========================
# Simple orientation helper (head–feet vertical span)
# =========================
mp_pose = mp.solutions.pose

def decide_rotation_from_head_feet(first_frame_bgr, pose_obj, min_vis=0.5, span_frac_thresh=0.1):
    """
    Use initial frame + pose to decide if we should rotate 90 degrees.
    If head–feet vertical span is small (< span_frac_thresh * height), we assume video is sideways
    and return cv2.ROTATE_90_CLOCKWISE. Otherwise, return None (no rotation).
    """
    h, w = first_frame_bgr.shape[:2]
    rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
    res = pose_obj.process(rgb)
    lm = res.pose_landmarks.landmark if res.pose_landmarks else None
    if not lm:
        print("[ORIENTATION] No pose found on first frame; no auto-rotation.")
        return None

    def vis(i):
        return (i < len(lm)) and (lm[i].visibility >= min_vis)

    nose_idx = mp_pose.PoseLandmark.NOSE.value
    la_idx   = mp_pose.PoseLandmark.LEFT_ANKLE.value
    ra_idx   = mp_pose.PoseLandmark.RIGHT_ANKLE.value

    if not vis(nose_idx):
        print("[ORIENTATION] Nose not visible; no auto-rotation.")
        return None

    head_y = lm[nose_idx].y * h
    foot_ys = []
    if vis(la_idx): foot_ys.append(lm[la_idx].y * h)
    if vis(ra_idx): foot_ys.append(lm[ra_idx].y * h)
    if not foot_ys:
        print("[ORIENTATION] Ankles not visible; no auto-rotation.")
        return None

    feet_y = float(np.median(foot_ys))
    vertical_span = feet_y - head_y

    if vertical_span <= 0:
        print(f"[ORIENTATION] vertical_span <= 0 (span={vertical_span:.1f}); no auto-rotation.")
        return None

    thresh_px = span_frac_thresh * h
    print(f"[ORIENTATION] span={vertical_span:.1f}px, thresh={thresh_px:.1f}px (h={h})")

    if vertical_span < thresh_px:
        print("[ORIENTATION] Span small => rotating 90° clockwise.")
        return cv2.ROTATE_90_CLOCKWISE
    else:
        print("[ORIENTATION] Span large => no rotation.")
        return None

# =========================
# Edge detection
# =========================
def detect_edges_hough(frame_bgr,
                       angle_max_deg=ANGLE_MAX_DEG_INIT,
                       min_len=MIN_LINE_LENGTH_INIT,
                       max_gap=MAX_LINE_GAP_INIT,
                       hough_thr=HOUGH_THRESH_INIT,
                       canny_low=CANNY_LOW_INIT,
                       canny_high=CANNY_HIGH_INIT,
                       shave_bottom=ROI_SHAVE_BOTTOM):
    """Return (edges_y_sorted, nearest_y, debug_img) for one frame."""
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
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if dx == 0:
                continue
            slope = dy / dx
            if abs(slope) <= max_slope and abs(x2 - x1) >= min_len:
                ymid = int((y1 + y2) // 2)
                y_list.append(ymid)
                cv2.line(debug, (x1, y1), (x2, y2), (0, 160, 0), 1)

    if not y_list:
        return [], None, debug

    y_list.sort()
    merged, cur = [], [y_list[0]]
    for y in y_list[1:]:
        if abs(y - cur[-1]) <= Y_CLUSTER_BIN:
            cur.append(y)
        else:
            merged.append(int(np.median(cur))); cur = [y]
    merged.append(int(np.median(cur)))
    merged = sorted(set(merged))
    nearest_y = max(merged)

    for y in merged:
        color = (0,255,0) if y == nearest_y else (0,200,0)
        thick = 3 if y == nearest_y else 2
        cv2.line(debug, (0, y), (w, y), color, thick)

    return merged, nearest_y, debug

def detect_edges_first_robust(frame_bgr):
    ylist, near, dbg = detect_edges_hough(frame_bgr)
    if near is not None:
        return ylist, near, ("strict", dbg)
    for (ang, mlen, gap, hthr, cl, ch, shave) in RELAX_SCHEDULE:
        ylist, near, dbg = detect_edges_hough(frame_bgr, ang, mlen, gap, hthr, cl, ch, shave)
        if near is not None:
            return ylist, near, ("relaxed", dbg)
    return [], None, ("none", frame_bgr.copy())

# =========================
# Manual line selection
# =========================
_manual_y = None
def _on_mouse(event, x, y, flags, param):
    global _manual_y
    if event == cv2.EVENT_LBUTTONDOWN:
        _manual_y = y

# =========================
# Main
# =========================
def analyze(video_path):
    global _manual_y
    pose = create_pose()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ok, first = cap.read()
    if not ok:
        cap.release(); raise RuntimeError("Could not read first frame.")

    # ---------- Orientation from first frame (simple head–feet span) ----------
    rotate_code = decide_rotation_from_head_feet(first, pose)
    if rotate_code is not None:
        first = cv2.rotate(first, rotate_code)

    # Edge detection on first (possibly rotated) frame
    edges_y, near_y, (tier, dbg) = detect_edges_first_robust(first)
    print(f"[Edges] tier={tier}, count={len(edges_y)}, nearest_y={near_y}")
    if SAVE_DEBUG: cv2.imwrite("first_frame_edges.png", dbg)

    v_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if v_fps < 1: v_fps = 30.0
    frame_interval = max(1, int(round(v_fps / ANALYZE_MAX_FPS)))
    eff_fps = (v_fps / frame_interval) if frame_interval > 0 else v_fps

    # Landmarks
    L_ANK, R_ANK = 27, 28
    L_HIP, R_HIP = 23, 24

    # Histories
    feet_y_hist = deque(maxlen=240)   # feet line (median of ankles)
    hip_y_hist  = deque(maxlen=240)   # for start detection

    # Baselines
    baseline_frames = int(BASELINE_SEC * max(1.0, eff_fps))
    feet_y_samples = []

    # State
    started = False; start_time = None
    finished = False; finish_time = None
    dwell = 0
    y_edge_smooth = float(near_y) if near_y is not None else np.nan
    first_ts = None
    frame_idx = 1

    def armed_now(y_feet_med, y_edge_smooth):
        return (y_feet_med - y_edge_smooth) >= CROSS_MARGIN_PX

    # UI
    init_window()
    cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)
    print("Keys: q=quit, f=fullscreen, e=redetect edges, m=manual line (click), s=force start")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # Apply same rotation to all frames if chosen
        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        if frame_idx % frame_interval != 0: continue

        t_abs = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if first_ts is None: first_ts = t_abs
        t_rel = t_abs - first_ts
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        lm = res.pose_landmarks.landmark if res.pose_landmarks else None

        # ---- Points ----
        ankL = ankR = (np.nan, np.nan)
        hip_mid = (np.nan, np.nan)
        if lm:
            if visible(lm, L_ANK): ankL = get_xy(lm, L_ANK, w, h)
            if visible(lm, R_ANK): ankR = get_xy(lm, R_ANK, w, h)
            if visible(lm, L_HIP) and visible(lm, R_HIP):
                lhip = get_xy(lm, L_HIP, w, h); rhip = get_xy(lm, R_HIP, w, h)
                hip_mid = ((lhip[0]+rhip[0])*0.5, (lhip[1]+rhip[1])*0.5)

        feet_candidates = [v for v in [ankL[1], ankR[1]] if not np.isnan(v)]
        y_feet = float(np.median(feet_candidates)) if feet_candidates else np.nan

        # Histories (just for smoothing via rolling median)
        push(feet_y_hist, y_feet, feet_y_hist.maxlen)
        push(hip_y_hist,  hip_mid[1], hip_y_hist.maxlen)
        y_feet_med = median_last(feet_y_hist, 5)

        # Baseline collection for start detection
        if len(feet_y_samples) < baseline_frames and not np.isnan(y_feet_med):
            feet_y_samples.append(y_feet_med)
        feet_y_base = float(np.median(feet_y_samples)) if feet_y_samples else np.nan

        # Start detection
        if not started and (frame_idx >= START_MIN_FRAMES) and (len(feet_y_samples) >= 3):
            ys = [y for y in list(hip_y_hist)[-6:] if not np.isnan(y)]
            hip_vel = 0.0
            if len(ys) >= 2:
                dt = (len(ys)-1) / max(1.0, eff_fps)
                hip_vel = abs((ys[-2] - ys[-1]) / dt) if dt > 0 else 0.0
            feet_delta_ok = (not np.isnan(feet_y_base) and not np.isnan(y_feet_med)
                             and (y_feet_med - feet_y_base) >= START_Y_DELTA_PX)
            if (hip_vel > HIP_UP_VEL_PX_S) or feet_delta_ok:
                started = True; start_time = t_rel
                print(f"[START] t={t_rel:.2f}s (|hip_vel|={hip_vel:.1f} px/s, feet_delta={'OK' if feet_delta_ok else 'no'})")

        # Hotkeys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'): toggle_fullscreen()
        elif key == ord('e'):
            edges_y, near_y_now, (tier2, dbg2) = detect_edges_first_robust(frame)
            if near_y_now is not None:
                y_edge_smooth = near_y_now if np.isnan(y_edge_smooth) else \
                                EDGE_EMA_ALPHA*near_y_now + (1-EDGE_EMA_ALPHA)*y_edge_smooth
            print(f"[Edges] re-detect tier={tier2}, count={len(edges_y)}, nearest_y={near_y_now}, y_edge_smooth≈{int(y_edge_smooth) if not np.isnan(y_edge_smooth) else 'n/a'}")
            if SAVE_DEBUG: cv2.imwrite("redetect_edges.png", dbg2)
        elif key == ord('m'):
            _manual_y = None
            cv2.setMouseCallback(WINDOW_NAME, lambda e,x,y,f,p: _on_mouse(e,x,y,f,p))
            print("Click once to set the finish line (y). Press ESC to cancel.")
            while True:
                preview = frame.copy()
                if _manual_y is not None:
                    cv2.line(preview, (0,_manual_y), (w,_manual_y), (0,255,255), 3)
                disp = resize_keep_aspect(preview, SCREEN_W, SCREEN_H) if FIT_TO_SCREEN else preview
                cv2.imshow(WINDOW_NAME, disp)
                k2 = cv2.waitKey(10) & 0xFF
                if k2 == 27:
                    _manual_y = None
                    cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)
                    print("Manual selection cancelled.")
                    break
                if _manual_y is not None:
                    near_y = _manual_y
                    y_edge_smooth = near_y if np.isnan(y_edge_smooth) else \
                                    EDGE_EMA_ALPHA*near_y + (1-EDGE_EMA_ALPHA)*y_edge_smooth
                    cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)
                    print(f"Manual finish y set to {near_y}")
                    break
        elif key == ord('s') and START_MANUAL_KEY:
            if not started:
                started = True; start_time = t_rel
                print(f"[START] forced manually at t={t_rel:.2f}s")
        elif key == ord('q'):
            break

        # Keep edge stable
        if near_y is not None and not np.isnan(y_edge_smooth):
            y_edge_smooth = EDGE_EMA_ALPHA*near_y + (1-EDGE_EMA_ALPHA)*y_edge_smooth

        # ---- FINISH: if armed => stop immediately ----
        if started and not finished and not np.isnan(y_feet_med) and not np.isnan(y_edge_smooth):
            is_armed = armed_now(y_feet_med, y_edge_smooth)
            if is_armed:
                dwell += 1
                if dwell >= DWELL_FRAMES:
                    finished = True
                    finish_time = t_rel
                    print(f"[FINISH] t={finish_time:.2f}s  (armed with margin >= {CROSS_MARGIN_PX}px)")
            else:
                dwell = max(0, dwell - 1)

        # ---- Draw overlay ----
        if res.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )

        # Edge (green)
        if not np.isnan(y_edge_smooth):
            y_edge_int = int(round(y_edge_smooth))
            cv2.line(frame, (0, y_edge_int), (w, y_edge_int), (0,255,0), 3)
            cv2.putText(frame, f"Nearest edge y~{y_edge_int}", (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Feet line (blue)
        if not np.isnan(y_feet_med):
            y_feet_int = int(round(y_feet_med))
            cv2.line(frame, (0, y_feet_int), (w, y_feet_int), (255,0,0), 2)
            cv2.putText(frame, "Feet line", (20, max(20, y_feet_int-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # Status/time/armed/distance
        status = "Waiting start"
        if started and not finished: status = "Climbing"
        if finished: status = "Done"

        d_txt = "n/a"; armed_flag = False
        if not np.isnan(y_feet_med) and not np.isnan(y_edge_smooth):
            d_val = y_feet_med - y_edge_smooth
            d_txt = f"{d_val:.1f}px"
            armed_flag = d_val >= CROSS_MARGIN_PX

        info_y = h - 78
        cv2.putText(frame, f"Status: {status}", (20, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,0), 2)
        cv2.putText(frame, f"d_feet = {d_txt}   dwell={dwell}/{DWELL_FRAMES}",
                    (20, info_y+26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
        arm_color = (0,165,255) if armed_flag else (180,180,180)
        cv2.putText(frame, f"armed: {'YES' if armed_flag else 'no'}  (margin >= {CROSS_MARGIN_PX}px)",
                    (20, info_y-24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, arm_color, 2)
        cv2.putText(frame, f"t = {t_rel:.2f}s", (20, info_y+52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        disp = resize_keep_aspect(frame, SCREEN_W, SCREEN_H) if FIT_TO_SCREEN else frame
        cv2.imshow(WINDOW_NAME, disp)

        if finished: break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    # =========================
    # Scoring
    # =========================
    print("\n=== RESULTS (Feet-line gate — instant) ===")
    if start_time is None:
        print("Start not detected.")
        print("Score: 0 (unable)")
        return
    if finish_time is None:
        print(f"Started at {start_time:.2f}s, but finish not reached before video end.")
        print("Score: 0 (unable)")
        return

    total = finish_time - start_time
    print(f"Start : {start_time:.2f}s")
    print(f"Finish: {finish_time:.2f}s")
    print(f"Total : {total:.2f}s")

    score, band = 0, "unable"
    if total < 5.0:
        score, band = 4, "< 5.0 s"
    elif 5.1 <= total <= 10.0:
        score, band = 3, "5.1–10.0 s"
    elif 10.1 <= total <= 15.0:
        score, band = 2, "10.1–15.0 s"
    elif total > 15.0:
        score, band = 1, "> 15.0 s"
    else:
        if 5.0 <= total < 5.1:
            score, band = 3, "≈ 5.0–5.1 s"
        elif 10.0 <= total < 10.1:
            score, band = 2, "≈ 10.0–10.1 s"
        elif 15.0 <= total < 15.1:
            score, band = 1, "≈ 15.0–15.1 s"

    print(f"Score: {score} (band: {band})")

# =========================
# Entry
# =========================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analyze(os.path.join(script_dir, VIDEO_FILENAME))
