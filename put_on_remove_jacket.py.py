import os, glob, argparse
import cv2
import numpy as np
from collections import deque
from skimage.metrics import structural_similarity as ssim
import mediapipe as mp

# =========================
# User files (same folder)
# =========================
DEFAULT_VIDEO_FILE     = "recording.mp4"   # video to analyze

# =========================
# Settings
# =========================
ANALYZE_MAX_FPS   = 12           # analyze slowly for stability
MIN_SEG_PX        = 40           # min elbow↔wrist length for reliable crop
BAND_FRAC         = 0.25         # forearm band thickness (fraction of length)
FOREARM_SIZE      = (140, 80)    # (w,h) normalized forearm crop
TORSO_SIZE        = (224, 300)   # (w,h) normalized torso crop

# ---- NEW: Build baseline from initial part of *video* ----
BASELINE_SEC          = 1      # use first ~2.5 s to build baseline
BASELINE_MIN_FRAMES   = 8       # need at least this many good frames
BASELINE_EMA_ALPHA    = 0.25      # EMA to accumulate baseline crops

# Auto-thresholds from a single image via tiny synthetic jitter (for PUT-ON only)
AUTO_SSIM_THRESHOLDS = True
SYN_JITTER_SAMPLES   = 8
JITTER_CONTRAST      = 0.04
JITTER_BRIGHT        = 0.02
JITTER_NOISE_STD     = 0.005

# Fallback thresholds for PUT-ON (vs baseline)
F_SSIME_ON_DEFAULT   = 0.80
T_SSIME_ON_DEFAULT   = 0.86

# Stability windows (seconds)
WIN_SEC         = 0.5
STABILITY_SEC   = 1.0
COND_HOLD_FRAC  = 0.75

# Posture bands (wrists slightly below hips)
REST_ZONE_HI_PUT    = 0.18
REST_ZONE_HI_REMOVE = 0.22

# ON-template capture after PUT-ON confirm
ON_CAPTURE_SEC  = 0.8

# ===== REMOVE: ON-only rule =====
# If similarity-to-ON gets BELOW these, we call it "removed" (with posture).
ON_DIFF_TH_FOREARM = 0.88
ON_DIFF_TH_TORSO   = 0.90

DRAW_THICK = 2
mp_pose = mp.solutions.pose


# =========================
# Small utilities
# =========================
def find_file_or_fallback(folder, preferred, patterns):
    pref = os.path.join(folder, preferred)
    if os.path.isfile(pref): return pref
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(folder, pat)))
        if hits: return hits[0]
    return None

def torso_box(lm, w, h):
    get = lambda i: (int(lm[i].x*w), int(lm[i].y*h))
    L_SH = get(mp_pose.PoseLandmark.LEFT_SHOULDER)
    R_SH = get(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    L_HP = get(mp_pose.PoseLandmark.LEFT_HIP)
    R_HP = get(mp_pose.PoseLandmark.RIGHT_HIP)
    x1 = min(L_SH[0], R_SH[0], L_HP[0], R_HP[0])
    x2 = max(L_SH[0], R_SH[0], L_HP[0], R_HP[0])
    y1 = min(L_SH[1], R_SH[1])
    y2 = max(L_HP[1], R_HP[1])
    pad_x = int(0.12 * (x2-x1+1)); pad_y = int(0.12 * (y2-y1+1))
    x1 = max(0, x1 - pad_x); x2 = x2 + pad_x
    y1 = max(0, y1 - pad_y); y2 = y2 + pad_y
    return (x1, y1, x2, y2)

def poly_from_segment(a, b, thickness_px):
    ax, ay = a; bx, by = b
    v = np.array([bx-ax, by-ay], dtype=np.float32)
    L = np.linalg.norm(v) + 1e-6
    n = v / L
    p = np.array([-n[1], n[0]], dtype=np.float32)
    t = thickness_px
    return np.array([
        [ax + p[0]*t, ay + p[1]*t],
        [ax - p[0]*t, ay - p[1]*t],
        [bx - p[0]*t, by - p[1]*t],
        [bx + p[0]*t, by + p[1]*t],
    ], dtype=np.int32)

def crop_masked(gray, polygon, out_size):
    if polygon is None: return None
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    x, y, ww, hh = cv2.boundingRect(polygon)
    x2, y2 = x+ww, y+hh
    x, y, x2, y2 = max(0,x), max(0,y), min(w,x2), min(h,y2)
    roi = gray[y:y2, x:x2]
    if roi.size == 0: return None
    mroi = mask[y:y2, x:x2]
    fore = cv2.bitwise_and(roi, roi, mask=mroi)
    return cv2.resize(fore, out_size, interpolation=cv2.INTER_AREA)

def crop_rect(gray_or_bgr, rect, out_size):
    x1, y1, x2, y2 = rect
    roi = gray_or_bgr[y1:y2, x1:x2]
    if roi.size == 0: return None
    return cv2.resize(roi, out_size, interpolation=cv2.INTER_AREA)

def wrists_slightly_below_hips(lm, w, h, band_frac):
    get = lambda i: (int(lm[i].x*w), int(lm[i].y*h))
    L_WR = get(mp_pose.PoseLandmark.LEFT_WRIST)
    R_WR = get(mp_pose.PoseLandmark.RIGHT_WRIST)
    L_SH = get(mp_pose.PoseLandmark.LEFT_SHOULDER)
    R_SH = get(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    L_HP = get(mp_pose.PoseLandmark.LEFT_HIP)
    R_HP = get(mp_pose.PoseLandmark.RIGHT_HIP)
    sh_y  = 0.5*(L_SH[1]+R_SH[1])
    hip_y = 0.5*(L_HP[1]+R_HP[1])
    torso_h = max(10, int(hip_y - sh_y))
    lo = hip_y
    hi = hip_y + band_frac * torso_h
    cond_L = (L_WR[1] >= lo) and (L_WR[1] <= hi)
    cond_R = (R_WR[1] >= lo) and (R_WR[1] <= hi)
    return cond_L and cond_R

def any_wrist_above_hip(lm, w, h):
    get = lambda i: (int(lm[i].x*w), int(lm[i].y*h))
    L_WR = get(mp_pose.PoseLandmark.LEFT_WRIST)
    R_WR = get(mp_pose.PoseLandmark.RIGHT_WRIST)
    L_HP = get(mp_pose.PoseLandmark.LEFT_HIP)
    R_HP = get(mp_pose.PoseLandmark.RIGHT_HIP)
    hip_y = 0.5*(L_HP[1]+R_HP[1])
    return (L_WR[1] < hip_y) or (R_WR[1] < hip_y)

def normalize_to_ref(img01, ref01, eps=1e-6):
    m, s  = float(img01.mean()), float(img01.std())
    mr, sr = float(ref01.mean()), float(ref01.std())
    if s < eps: return img01
    out = (img01 - m) * (sr / max(s, eps)) + mr
    return np.clip(out, 0.0, 1.0)

def jitter_samples(base, n=6, c=0.04, b=0.02, noise=0.005):
    out = []
    for _ in range(n):
        g = base.copy()
        alpha = 1.0 + np.random.uniform(-c, c)    # contrast
        beta  = np.random.uniform(-b, b)          # brightness
        g = np.clip(g*alpha + beta, 0, 1)
        g = g + np.random.normal(0, noise, g.shape)
        out.append(np.clip(g, 0, 1).astype(np.float32))
    return out


# =========================
# NEW: Baseline from initial video segment
# =========================
def load_baseline_from_video_initial(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for baseline: {video_path}")
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Failed to read first video frame.")

    h, w = first.shape[:2]
    fps_src = cap.get(cv2.CAP_PROP_FPS); fps_src = fps_src if fps_src and fps_src > 1 else 30.0
    max_frames = int(BASELINE_SEC * fps_src)

    pose = mp_pose.Pose(model_complexity=1, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    L_base = None; R_base = None; T_base = None
    used = 0

    for i in range(max_frames):
        ok, frame = cap.read()
        if not ok: break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks: continue
        lm = res.pose_landmarks.landmark
        get = lambda i: (int(lm[i].x*w), int(lm[i].y*h))
        L_EL, R_EL = get(mp_pose.PoseLandmark.LEFT_ELBOW),  get(mp_pose.PoseLandmark.RIGHT_ELBOW)
        L_WR, R_WR = get(mp_pose.PoseLandmark.LEFT_WRIST),  get(mp_pose.PoseLandmark.RIGHT_WRIST)
        lenL = int(np.linalg.norm(np.array(L_WR) - np.array(L_EL)))
        lenR = int(np.linalg.norm(np.array(R_WR) - np.array(R_EL)))

        # Prefer neutral/rest posture for baseline
        if lenL < MIN_SEG_PX or lenR < MIN_SEG_PX: continue
        if not wrists_slightly_below_hips(lm, w, h, REST_ZONE_HI_PUT): continue

        tL = max(6, int(BAND_FRAC * lenL)); tR = max(6, int(BAND_FRAC * lenR))
        L_poly = poly_from_segment(L_EL, L_WR, tL)
        R_poly = poly_from_segment(R_EL, R_WR, tR)

        L_crop = crop_masked(g, L_poly, FOREARM_SIZE)
        R_crop = crop_masked(g, R_poly, FOREARM_SIZE)
        box = torso_box(lm, w, h)
        T_crop = crop_rect(g, box, TORSO_SIZE)

        if L_crop is None or R_crop is None or T_crop is None:
            continue

        Lf = L_crop.astype(np.float32)/255.0
        Rf = R_crop.astype(np.float32)/255.0
        Tf = T_crop.astype(np.float32)/255.0

        # EMA accumulate
        L_base = Lf if L_base is None else (1.0-BASELINE_EMA_ALPHA)*L_base + BASELINE_EMA_ALPHA*Lf
        R_base = Rf if R_base is None else (1.0-BASELINE_EMA_ALPHA)*R_base + BASELINE_EMA_ALPHA*Rf
        T_base = Tf if T_base is None else (1.0-BASELINE_EMA_ALPHA)*T_base + BASELINE_EMA_ALPHA*Tf
        used += 1

    cap.release()

    if used < BASELINE_MIN_FRAMES or L_base is None or R_base is None or T_base is None:
        raise RuntimeError(
            f"Could not build a reliable baseline from the first {BASELINE_SEC}s. "
            f"Got {used} good frames. Make sure the subject starts in a neutral stance with arms at rest."
        )

    # Torso stats
    T_mean = float(T_base.mean()); T_std = float(T_base.std())
    print("\n=== TORSO BASELINE DETAILS (from video) ===")
    print(f"Frames used: {used}")
    print(f"Torso crop size: {TORSO_SIZE[0]}x{TORSO_SIZE[1]}")
    print(f"Torso gray mean: {T_mean:.3f}")
    print(f"Torso gray std:  {T_std:.3f}")

    # Auto thresholds for PUT-ON only (vs baseline)
    def auto_pair_thresholds(base, name, default_on):
        if not AUTO_SSIM_THRESHOLDS:
            print(f"[Default {name}] TH_ON={default_on:.3f}")
            return default_on
        samples = jitter_samples(base, SYN_JITTER_SAMPLES, JITTER_CONTRAST, JITTER_BRIGHT, JITTER_NOISE_STD)
        vals = [ssim(base, s, data_range=1.0) for s in samples]
        mean_ssim = float(np.mean(vals))
        std_ssim  = float(np.clip(np.std(vals), 1e-3, 0.2))
        th_on  = max(0.0, min(1.0, mean_ssim - 3.0*std_ssim))
        print(f"[Auto {name}] mean={mean_ssim:.3f}, std={std_ssim:.3f} -> TH_ON={th_on:.3f}")
        return th_on

    F_TH_ON_L = auto_pair_thresholds(L_base, "FOREARM-L", F_SSIME_ON_DEFAULT)
    F_TH_ON_R = auto_pair_thresholds(R_base, "FOREARM-R", F_SSIME_ON_DEFAULT)
    F_TH_ON   = min(F_TH_ON_L, F_TH_ON_R)
    T_TH_ON   = auto_pair_thresholds(T_base, "TORSO", T_SSIME_ON_DEFAULT)

    thresholds = {"F_TH_ON": F_TH_ON, "T_TH_ON": T_TH_ON}
    print(f"[PUT-ON THRESHOLDS] F_ON={F_TH_ON:.3f}  T_ON={T_TH_ON:.3f}")

    return L_base, R_base, T_base, thresholds


# =========================
# Analysis (PUT-ON vs baseline; REMOVE vs ON only)
# =========================
def analyze_video_with_files(video_path, L_base, R_base, T_base, thresholds):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Failed to read first video frame.")

    h, w = first.shape[:2]
    fps_src = cap.get(cv2.CAP_PROP_FPS); fps_src = fps_src if fps_src and fps_src>1 else 30.0
    fps_proc = min(fps_src, ANALYZE_MAX_FPS)
    frame_delay_ms = int(1000.0 / fps_proc)

    pose = mp_pose.Pose(model_complexity=1, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    win_len = max(5, int(WIN_SEC * fps_proc))
    # To-baseline SSIM (for PUT-ON)
    L_ssimB = deque(maxlen=win_len); R_ssimB = deque(maxlen=win_len); T_ssimB = deque(maxlen=win_len)
    # To-ON SSIM (for REMOVE)
    L_ssimO = deque(maxlen=win_len); R_ssimO = deque(maxlen=win_len); T_ssimO = deque(maxlen=win_len)

    put_series = deque(maxlen=win_len)
    rem_series = deque(maxlen=win_len)

    # ON templates captured after PUT ON
    L_on = None; R_on = None; T_on = None
    on_collecting = False
    on_collect_end_t = None

    # Start timing trigger
    start_time = None
    start_hud_show_until = 0.0

    state = "OFF"
    put_time = None; rem_time = None
    last_toggle_t = 0.0

    # restart stream for analysis from the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("== Analyzing (press 'q' to stop)…")
    while True:
        ok, frame = cap.read()
        if not ok: break

        # Video time (seconds)
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # per-frame SSIMs
        Lb = 1.0; Rb = 1.0; Tb = 1.0
        Lo = 1.0; Ro = 1.0; To = 1.0
        posture_ok_put = False
        posture_ok_remove = False

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            # Start timer when any wrist above hip
            if start_time is None and any_wrist_above_hip(lm, w, h):
                start_time = t
                start_hud_show_until = t + 2.0  # show “timer started” for 2s

            get = lambda i: (int(lm[i].x*w), int(lm[i].y*h))
            L_EL, R_EL = get(mp_pose.PoseLandmark.LEFT_ELBOW),  get(mp_pose.PoseLandmark.RIGHT_ELBOW)
            L_WR, R_WR = get(mp_pose.PoseLandmark.LEFT_WRIST),  get(mp_pose.PoseLandmark.RIGHT_WRIST)

            posture_ok_put    = wrists_slightly_below_hips(lm, w, h, REST_ZONE_HI_PUT)
            posture_ok_remove = wrists_slightly_below_hips(lm, w, h, REST_ZONE_HI_REMOVE)

            # Forearms
            lenL = int(np.linalg.norm(np.array(L_WR) - np.array(L_EL)))
            lenR = int(np.linalg.norm(np.array(R_WR) - np.array(R_EL)))
            L_poly = R_poly = None
            if lenL >= MIN_SEG_PX and lenR >= MIN_SEG_PX:
                tL = max(6, int(BAND_FRAC * lenL)); tR = max(6, int(BAND_FRAC * lenR))
                L_poly = poly_from_segment(L_EL, L_WR, tL)
                R_poly = poly_from_segment(R_EL, R_WR, tR)
                L_crop = crop_masked(g, L_poly, FOREARM_SIZE)
                R_crop = crop_masked(g, R_poly, FOREARM_SIZE)
                if L_crop is not None:
                    Lc = L_crop.astype(np.float32)/255.0
                    LcB = normalize_to_ref(Lc, L_base); Lb = ssim(L_base, LcB, data_range=1.0)
                    if L_on is not None:
                        LcO = normalize_to_ref(Lc, L_on);  Lo = ssim(L_on, LcO, data_range=1.0)
                if R_crop is not None:
                    Rc = R_crop.astype(np.float32)/255.0
                    RcB = normalize_to_ref(Rc, R_base); Rb = ssim(R_base, RcB, data_range=1.0)
                    if R_on is not None:
                        RcO = normalize_to_ref(Rc, R_on);  Ro = ssim(R_on, RcO, data_range=1.0)
                if L_poly is not None: cv2.polylines(frame, [L_poly], True, (0,255,255), DRAW_THICK)
                if R_poly is not None: cv2.polylines(frame, [R_poly], True, (0,255,255), DRAW_THICK)

            # Torso
            box = torso_box(lm, w, h)
            T_crop = crop_rect(g, box, TORSO_SIZE)
            if T_crop is not None:
                Tc = T_crop.astype(np.float32)/255.0
                TcB = normalize_to_ref(Tc, T_base); Tb = ssim(T_base, TcB, data_range=1.0)
                if T_on is not None:
                    TcO = normalize_to_ref(Tc, T_on); To = ssim(T_on, TcO, data_range=1.0)
            cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), (0,255,0), DRAW_THICK)

        # Update windows
        win_len = max(5, int(WIN_SEC * ANAlYZE_MAX_FPS)) if False else None  # kept above
        L_ssimB.append(Lb); R_ssimB.append(Rb); T_ssimB.append(Tb)
        if L_on is not None: L_ssimO.append(Lo)
        if R_on is not None: R_ssimO.append(Ro)
        if T_on is not None: T_ssimO.append(To)

        # Rolling stats
        L_medB = float(np.median(L_ssimB)); R_medB = float(np.median(R_ssimB)); T_medB = float(np.median(T_ssimB))
        L_medO = float(np.median(L_ssimO)) if len(L_ssimO)>0 else 1.0
        R_medO = float(np.median(R_ssimO)) if len(R_ssimO)>0 else 1.0
        T_medO = float(np.median(T_ssimO)) if len(T_ssimO)>0 else 1.0
        L_minO = float(np.min(L_ssimO)) if len(L_ssimO)>0 else 1.0
        R_minO = float(np.min(R_ssimO)) if len(R_ssimO)>0 else 1.0
        T_minO = float(np.min(T_ssimO)) if len(T_ssimO)>0 else 1.0

        # PUT-ON: baseline only
        F_TH_ON = thresholds["F_TH_ON"]; T_TH_ON = thresholds["T_TH_ON"]
        forearms_on  = (L_medB <= F_TH_ON and R_medB <= F_TH_ON)
        torso_on     = (T_medB <= T_TH_ON)
        put_combined = posture_ok_put and forearms_on and torso_on
        put_series.append(put_combined)
        put_hold = (np.mean(put_series) >= COND_HOLD_FRAC)

        # Confirm PUT-ON -> capture ON templates briefly
        if state == "OFF" and (t - last_toggle_t) >= STABILITY_SEC and put_hold:
            state = "ON"; put_time = t; last_toggle_t = t
            L_on = None; R_on = None; T_on = None
            on_collecting = True
            on_collect_end_t = t + ON_CAPTURE_SEC
            L_ssimO.clear(); R_ssimO.clear(); T_ssimO.clear()

        # Accumulate ON templates (EMA)
        if on_collecting and t <= on_collect_end_t and res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            get = lambda i: (int(lm[i].x*w), int(lm[i].y*h))
            L_EL, R_EL = get(mp_pose.PoseLandmark.LEFT_ELBOW),  get(mp_pose.PoseLandmark.RIGHT_ELBOW)
            L_WR, R_WR = get(mp_pose.PoseLandmark.LEFT_WRIST),  get(mp_pose.PoseLandmark.RIGHT_WRIST)
            lenL = int(np.linalg.norm(np.array(L_WR) - np.array(L_EL)))
            lenR = int(np.linalg.norm(np.array(R_WR) - np.array(R_EL)))
            if lenL >= MIN_SEG_PX and lenR >= MIN_SEG_PX:
                tL = max(6, int(BAND_FRAC * lenL)); tR = max(6, int(BAND_FRAC * lenR))
                L_poly = poly_from_segment(L_EL, L_WR, tL)
                R_poly = poly_from_segment(R_EL, R_WR, tR)
                L_crop = crop_masked(g, L_poly, FOREARM_SIZE)
                R_crop = crop_masked(g, R_poly, FOREARM_SIZE)
                box = torso_box(lm, w, h)
                T_crop = crop_rect(g, box, TORSO_SIZE)
                if L_crop is not None:
                    Lc = L_crop.astype(np.float32)/255.0
                    L_on = Lc if L_on is None else (0.85*L_on + 0.15*Lc)
                if R_crop is not None:
                    Rc = R_crop.astype(np.float32)/255.0
                    R_on = Rc if R_on is None else (0.85*R_on + 0.15*Rc)
                if T_crop is not None:
                    Tc = T_crop.astype(np.float32)/255.0
                    T_on = Tc if T_on is None else (0.85*T_on + 0.15*Tc)
        elif on_collecting and t > on_collect_end_t:
            on_collecting = False

        # ===== REMOVE: ON-only rule =====
        remove_looks_off = (L_minO <= ON_DIFF_TH_FOREARM and
                            R_minO <= ON_DIFF_TH_FOREARM and
                            T_minO <= ON_DIFF_TH_TORSO)

        rem_combined = (state == "ON") and posture_ok_remove and \
                       (L_on is not None and R_on is not None and T_on is not None) and \
                       remove_looks_off

        rem_series.append(rem_combined)
        rem_hold = (np.mean(rem_series) >= COND_HOLD_FRAC)

        if state == "ON" and (t - last_toggle_t) >= STABILITY_SEC and rem_hold:
            state = "OFF"; rem_time = t; last_toggle_t = t

        # -------- HUD --------
        def draw_text(img, text, pos, color, scale=0.6, thickness=2):
            x, y = pos
            cv2.putText(img, text, (x+1, y+1), 0, scale, (0,0,0), thickness+2)  # shadow
            cv2.putText(img, text, (x, y), 0, scale, color, thickness)

        if start_time is not None:
            elapsed = max(0.0, t - start_time)
            draw_text(frame, f"t={t:05.2f}s  ELAPSED(from 1st wrist>hip)={elapsed:05.2f}s  STATE: {'ON' if state=='ON' else 'OFF'}",
                      (18,40), (0,200,0) if state=='ON' else (0,0,255), scale=0.7)
        else:
            draw_text(frame, f"t={t:05.2f}s  STATE: {'ON' if state=='ON' else 'OFF'}  (waiting for first wrist > hip to start timer)",
                      (18,40), (0,200,0) if state=='ON' else (0,0,255), scale=0.7)

        if start_time is not None and t <= start_hud_show_until:
            draw_text(frame, "Timer started (wrist above hip detected)", (18,65), (0,255,255), scale=0.6)

        draw_text(frame, f"PUT gates: F_medB<={thresholds['F_TH_ON']:.2f} & T_medB<={thresholds['T_TH_ON']:.2f} | "
                         f"Posture PUT: {'YES' if posture_ok_put else 'NO'}",
                  (18,90), (255,255,0), scale=0.55, thickness=1)

        if 'L_on' in locals() and L_on is not None:
            draw_text(frame, f"ON ready. REMOVE uses SSIM-to-ON only.",
                      (18,115), (0,255,0), scale=0.6)
            draw_text(frame, f"Min SSIM-to-ON  L/R/T = {L_minO:.2f}/{R_minO:.2f}/{T_minO:.2f}  "
                             f"(need <= {ON_DIFF_TH_FOREARM:.2f}/{ON_DIFF_TH_FOREARM:.2f}/{ON_DIFF_TH_TORSO:.2f})",
                      (18,140), (0,255,255), scale=0.55, thickness=1)
        else:
            draw_text(frame, "Capturing ON templates… (after PUT-ON)", (18,115), (0,255,0), scale=0.6)

        draw_text(frame, f"Posture REMOVE: {'YES' if posture_ok_remove else 'NO'}",
                  (18,165), (200,200,200), scale=0.55, thickness=1)

        # draw regions
        if res.pose_landmarks:
            if 'L_poly' in locals() and L_poly is not None:
                cv2.polylines(frame, [L_poly], True, (0,255,255), DRAW_THICK)
            if 'R_poly' in locals() and R_poly is not None:
                cv2.polylines(frame, [R_poly], True, (0,255,255), DRAW_THICK)
            if 'box' in locals():
                cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), (0,255,0), DRAW_THICK)

        cv2.imshow("Jacket analysis (video-only baseline, ON-only remove)", frame)
        k = cv2.waitKey(frame_delay_ms) & 0xFF
        if k == ord('q'): break
        if 'put_time' in locals() and 'rem_time' in locals() and put_time is not None and rem_time is not None:
            break

    cap.release(); cv2.destroyAllWindows()

    # ===== SUMMARY =====
    print("\n===== SUMMARY =====")
    put_ok = ('put_time' in locals() and put_time is not None)
    rem_ok = ('rem_time' in locals() and rem_time is not None)
    if 'start_time' in locals() and start_time is None:
        print("Start trigger (first wrist above hip) was never observed.")
    print(f"Put ON confirmed:  {'YES' if put_ok else 'NO'}")
    print(f"Remove confirmed: {'YES' if (rem_ok and put_ok) else 'NO (requires successful put-on first)'}")
    if put_ok:
        if start_time is not None:
            print(f"Time to PUT ON: {put_time - start_time:.2f} s")
        else:
            print(f"Time to PUT ON: {put_time:.2f} s (no start trigger; absolute video time)")
    else:
        print("Time to PUT ON: not detected")

    if put_ok and rem_ok:
        if start_time is not None:
            print(f"Time to REMOVE: {rem_time - put_time:.2f} s")
            print(f"Total time (start → remove): {rem_time - start_time:.2f} s")
        else:
            print(f"Time to REMOVE: {rem_time - put_time:.2f} s")
            print(f"Total time (start → remove): {rem_time:.2f} s (no start trigger; absolute)")
    else:
        print("Time to REMOVE: not detected")
        print("Total time: not available")


# =========================
# Main
# =========================
def main():
    folder = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Analyze jacket put-on/remove using video-only (baseline from initial frames).")
    parser.add_argument("--video", "-v", default=None, help="Video filename.")
    args = parser.parse_args()

    video_path = os.path.join(folder, args.video) if args.video else \
        find_file_or_fallback(folder, DEFAULT_VIDEO_FILE, ["recording*.mp4", "*.mp4", "*.avi", "*.mov", "*.mkv"])

    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError("Video file not found. Put a video in this folder (e.g., recording.mp4).")

    print(f"[INFO] Using video: {os.path.basename(video_path)}")

    # Build baseline from the first BASELINE_SEC seconds of the video
    L_base, R_base, T_base, thresholds = load_baseline_from_video_initial(video_path)

    # Analyze full video (restart internally)
    analyze_video_with_files(video_path, L_base, R_base, T_base, thresholds)

if __name__ == "__main__":
    main()
