import cv2
import mediapipe as mp
import numpy as np

# -----------------------------
# SETTINGS
# -----------------------------
BASELINE_DURATION = 0.5         # seconds: initial no-jacket baseline

PUTON_DIFF_THRESH   = 30.0      # distance vs no-jacket baseline to say "changed"
REMOVE_DIFF_THRESH  = 30.0      # distance vs jacket-on baseline to say "changed"
STABLE_SEC          = 0.0       # need changed+stable for this many seconds
STABLE_DELTA_THRESH = 5.0       # max change vs previous frame to call it "stable"

REGION_NAMES = ['LS', 'RS', 'LF', 'RF']

# -----------------------------
# POSE SETUP
# -----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False
)

# -----------------------------
# REGION HELPERS
# -----------------------------
def _landmark_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

def _dist(p1, p2):
    return np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float))

def get_regions(frame, landmarks):
    """
    Returns:
        regions: dict name -> (x1, y1, x2, y2)
                 names: 'LS', 'RS', 'LF', 'RF'
        ok: bool
    """
    h, w, _ = frame.shape
    lm = landmarks.landmark

    try:
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        le = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
        re = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
        lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
    except Exception:
        return {}, False

    # require decent visibility
    for p in [ls, rs, le, re, lw, rw]:
        if p.visibility < 0.5:
            return {}, False

    LS = _landmark_px(ls, w, h)
    RS = _landmark_px(rs, w, h)
    LE = _landmark_px(le, w, h)
    RE = _landmark_px(re, w, h)
    LW = _landmark_px(lw, w, h)
    RW = _landmark_px(rw, w, h)

    shoulder_width    = max(_dist(LS, RS), 1.0)
    left_forearm_len  = max(_dist(LE, LW), 1.0)
    right_forearm_len = max(_dist(RE, RW), 1.0)

    sh_size = int(max(20, 0.3 * shoulder_width))

    def box_around(center, size):
        cx, cy = center
        x1 = int(max(0, cx - size // 2))
        y1 = int(max(0, cy - size // 2))
        x2 = int(min(w, cx + size // 2))
        y2 = int(min(h, cy + size // 2))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def mid(p1, p2):
        return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

    lf_center = mid(LE, LW)
    rf_center = mid(RE, RW)
    lf_size   = int(max(20, 0.5 * left_forearm_len))
    rf_size   = int(max(20, 0.5 * right_forearm_len))

    LS_box = box_around(LS, sh_size)
    RS_box = box_around(RS, sh_size)
    LF_box = box_around(lf_center, lf_size)
    RF_box = box_around(rf_center, rf_size)

    if None in [LS_box, RS_box, LF_box, RF_box]:
        return {}, False

    regions = {
        'LS': LS_box,
        'RS': RS_box,
        'LF': LF_box,
        'RF': RF_box
    }
    return regions, True

def compute_region_means(frame, regions):
    """
    Returns: dict name -> np.array([B,G,R], float)
    """
    means = {}
    for name, (x1, y1, x2, y2) in regions.items():
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        mean_bgr = patch.mean(axis=(0, 1))  # (B,G,R)
        means[name] = mean_bgr
    return means

def max_color_delta(means_a, means_b):
    """
    Max L2 distance over all 4 regions between two mean-color dicts.
    """
    if means_a is None or means_b is None:
        return float("inf")
    max_d = 0.0
    for k in REGION_NAMES:
        d = np.linalg.norm(means_a[k] - means_b[k])
        if d > max_d:
            max_d = d
    return max_d

# -----------------------------
# HEADLESS ANALYSIS
# -----------------------------
def analyze_jacket_video(video_name: str):
    """
    Analyze a video of "put on and remove a jacket".

    Returns a dict with:
        - success (bool)
        - error (str or None)
        - t_start, t_put_on, t_removed (seconds from video start)
        - put_on_time, remove_time, total_time (seconds)
        - score_0_4 (int, 0–4 based on total_time)
        - score_label (str)
    """
    results = {
        "success": False,
        "error": None,
        "t_start": None,
        "t_put_on": None,
        "t_removed": None,
        "put_on_time": None,
        "remove_time": None,
        "total_time": None,
        "score_0_4": None,
        "score_label": None
    }

    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        results["error"] = f"Could not open {video_name}."
        return results

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    frame_count = 0

    # Baseline accumulators (no jacket)
    baseline_sums   = {k: np.zeros(3, dtype=float) for k in REGION_NAMES}
    baseline_counts = {k: 0 for k in REGION_NAMES}
    baseline_no_jacket = None
    baseline_jacket_on = None
    baseline_ready = False

    state = "WAIT_PUT_ON"
    t_start = None
    t_put_on = None
    t_removed = None

    prev_means = None

    # stability tracking windows
    stable_start_puton   = None
    stable_acc_puton     = {k: np.zeros(3, dtype=float) for k in REGION_NAMES}
    stable_count_puton   = 0

    stable_start_remove  = None
    stable_acc_remove    = {k: np.zeros(3, dtype=float) for k in REGION_NAMES}
    stable_count_remove  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / fps

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            continue

        regions, ok = get_regions(frame, res.pose_landmarks)
        if not ok:
            continue

        means = compute_region_means(frame, regions)
        if means is None:
            continue

        # ---------------------------
        # 1) BUILD NO-JACKET BASELINE
        # ---------------------------
        if timestamp <= BASELINE_DURATION and state == "WAIT_PUT_ON":
            for k in REGION_NAMES:
                baseline_sums[k]   += means[k]
                baseline_counts[k] += 1
            prev_means = means
            continue

        if not baseline_ready:
            if any(baseline_counts[k] == 0 for k in baseline_counts):
                results["error"] = "Not enough baseline frames for all regions."
                cap.release()
                return results
            baseline_no_jacket = {
                k: baseline_sums[k] / baseline_counts[k] for k in REGION_NAMES
            }
            baseline_ready = True
            t_start = timestamp
            prev_means = means
            continue

        # ---------------------------
        # 2) STATE MACHINE with "changed AND stable" logic
        # ---------------------------
        # Per-frame color delta vs previous frame (for stability inside the window)
        frame_delta = max_color_delta(means, prev_means)
        prev_means = means

        if state == "WAIT_PUT_ON":
            # check if color has changed from NO-JACKET baseline in all 4 regions
            all_changed = True
            for k in REGION_NAMES:
                d = np.linalg.norm(means[k] - baseline_no_jacket[k])
                if d < PUTON_DIFF_THRESH:
                    all_changed = False
                    break

            # allow window to grow only when changed AND stable
            if all_changed and frame_delta < STABLE_DELTA_THRESH:
                if stable_start_puton is None:
                    stable_start_puton = timestamp
                    stable_acc_puton   = {kk: means[kk].copy() for kk in REGION_NAMES}
                    stable_count_puton = 1
                else:
                    for kk in REGION_NAMES:
                        stable_acc_puton[kk] += means[kk]
                    stable_count_puton += 1

                stable_duration = timestamp - stable_start_puton

                if stable_duration >= STABLE_SEC and t_put_on is None:
                    # confirmed put-on
                    t_put_on = stable_start_puton
                    baseline_jacket_on = {
                        kk: stable_acc_puton[kk] / stable_count_puton for kk in REGION_NAMES
                    }
                    state = "WAIT_REMOVE"
            else:
                stable_start_puton = None
                stable_count_puton = 0

        elif state == "WAIT_REMOVE":
            # check if color has changed from JACKET-ON baseline in all 4 regions
            all_changed = True
            for k in REGION_NAMES:
                d = np.linalg.norm(means[k] - baseline_jacket_on[k])
                if d < REMOVE_DIFF_THRESH:
                    all_changed = False
                    break

            if all_changed and frame_delta < STABLE_DELTA_THRESH:
                if stable_start_remove is None:
                    stable_start_remove = timestamp
                    stable_acc_remove   = {kk: means[kk].copy() for kk in REGION_NAMES}
                    stable_count_remove = 1
                else:
                    for kk in REGION_NAMES:
                        stable_acc_remove[kk] += means[kk]
                    stable_count_remove += 1

                stable_duration = timestamp - stable_start_remove

                if stable_duration >= STABLE_SEC and t_removed is None:
                    t_removed = stable_start_remove
                    state = "DONE"
                    # we can break after we detect removal
                    break
            else:
                stable_start_remove = None
                stable_count_remove = 0

        if state == "DONE":
            break

    cap.release()

    # ---------------------------
    # 3) FINAL TIMING + SCORING
    # ---------------------------
    if t_put_on is None or t_removed is None or t_start is None:
        results["error"] = "Could not detect full put-on / remove cycle."
        return results

    put_on_time = t_put_on - t_start
    remove_time = t_removed - t_put_on
    total_time = t_removed - t_start

    results["success"] = True
    results["t_start"] = float(t_start)
    results["t_put_on"] = float(t_put_on)
    results["t_removed"] = float(t_removed)
    results["put_on_time"] = float(put_on_time)
    results["remove_time"] = float(remove_time)
    results["total_time"] = float(total_time)

    # Scoring: put on + remove jacket (total_time)
    # <10 sec = 4
    # 10.1–15 sec = 3
    # 15.1–20 sec = 2
    # >20 sec = 1
    # unable = 0  (handled above when success=False)
    if total_time < 10.0:
        score = 4
        label = "<10 s"
    elif total_time <= 15.0:
        score = 3
        label = "10.1–15 s"
    elif total_time <= 20.0:
        score = 2
        label = "15.1–20 s"
    else:
        score = 1
        label = ">20 s"

    results["score"] = int(score)
    results["score_label"] = label

    return results
