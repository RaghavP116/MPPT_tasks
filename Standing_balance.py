import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define landmark indices for clarity
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value

LEFT_HEEL = mp_pose.PoseLandmark.LEFT_HEEL.value
RIGHT_HEEL = mp_pose.PoseLandmark.RIGHT_HEEL.value
LEFT_BIG_TOE = mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
RIGHT_BIG_TOE = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value

# Task-specific parameters
TASKS = ["Feet Together", "Semi-Tandem", "Full Tandem"]
TASK_DURATION = 10  # seconds per task
STABILITY_THRESHOLD = 0.05  # A small value in normalized coordinates

# =====================================================================
# RECORDING: single camera session, three 10s clips with 2s buffer each
# =====================================================================
def record_three_videos():
    """
    Opens the camera ONCE and records three 10s clips with 2s buffers:
    Feet Together, Semi-Tandem, Full Tandem.
    Saves files next to this script and returns the filenames.
    """
    files = [
        "feet_together_recording.mp4",
        "semi_tandem_recording.mp4",
        "full_tandem_recording.mp4"
    ]

    # --- Open camera once ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not access the camera. Ensure a webcam is connected and not in use.")

    # Best-effort resolution
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception:
        pass

    # FPS fallback logic
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 20.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    frame_size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    tasks_and_files = list(zip(TASKS, files))
    for task_name, filename in tasks_and_files:
        # ---- 2s "Get Ready" buffer (not recorded) ----
        buffer_start = time.time()
        while time.time() - buffer_start < 2.0:
            ok, frame = cap.read()
            if not ok:
                continue
            remaining = max(0.0, 2.0 - (time.time() - buffer_start))
            cv2.putText(frame, f"Get Ready: {task_name}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"Starting in: {remaining:.1f}s", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

        # ---- Start a new writer for this segment ----
        out = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        if not out.isOpened():
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError(f"Error: Could not open VideoWriter for {filename}")

        print(f"\n--- Recording: {task_name} -> {filename} ---")
        record_start = time.time()
        while time.time() - record_start < 10.0:
            ok, frame = cap.read()
            if not ok:
                continue
            elapsed = time.time() - record_start
            cv2.putText(frame, f"RECORDING: {task_name}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, f"Time: {elapsed:.1f}s / 10s", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            out.write(frame)
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        print(f"Saved: {filename}")
        # small pause to help some OSes flush the file handle
        cv2.waitKey(250)

    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.5)
    return files

# ======================================================
# Results/Scoring (UPDATED to your new rubric ONLY)
# ======================================================
def calculate_score(feet_together_time, semi_tandem_time, full_tandem_time):
    """
    New rubric:
    score, full tandem,   semi-tandem, feet together
    4,     10 seconds,    10 seconds,  10 seconds
    3,     3-9 seconds,   10 seconds,  10 seconds
    2,     0-2 seconds,   10 seconds,  10 seconds
    1,     0 seconds,     0-9 seconds, 10 seconds
    0,     0 seconds,     0-9 seconds, 0-9 seconds

    Uses small tolerance EPS to treat ~10s as 10s and ~0s as 0s.
    """
    EPS = 0.25

    def is_10s(x): return x >= (10.0 - EPS)
    def is_0s(x):  return x <= EPS
    def in_range(x, lo, hi): return (x >= (lo - EPS)) and (x <= (hi + EPS))

    # 4: all ~10s
    if is_10s(feet_together_time) and is_10s(semi_tandem_time) and is_10s(full_tandem_time):
        return 4
    # 3: full 3–9s, semi 10s, feet 10s
    if is_10s(feet_together_time) and is_10s(semi_tandem_time) and in_range(full_tandem_time, 3.0, 9.0):
        return 3
    # 2: full 0–2s, semi 10s, feet 10s
    if is_10s(feet_together_time) and is_10s(semi_tandem_time) and in_range(full_tandem_time, 0.0, 2.0):
        return 2
    # 1: full 0s, semi 0–9s, feet 10s
    if is_10s(feet_together_time) and in_range(semi_tandem_time, 0.0, 9.0) and is_0s(full_tandem_time):
        return 1
    # 0: full 0s, semi 0–9s, feet 0–9s
    if in_range(semi_tandem_time, 0.0, 9.0) and in_range(feet_together_time, 0.0, 9.0) and is_0s(full_tandem_time):
        return 0

    # Conservative fallback
    return 0

# --- Core Logic Functions (Foot Position Checks) ---
def check_feet_together(landmarks):
    """Checks if feet are parallel and touching."""
    lh_x = landmarks[LEFT_HEEL].x
    rh_x = landmarks[RIGHT_HEEL].x
    lb_x = landmarks[LEFT_BIG_TOE].x
    rb_x = landmarks[RIGHT_BIG_TOE].x
    if abs(lh_x - rh_x) < STABILITY_THRESHOLD and abs(lb_x - rb_x) < STABILITY_THRESHOLD:
        return True
    return False

def check_semi_tandem(landmarks):
    """Checks if the front heel is aligned with the back toe, and the feet are parallel."""
    lh_y = landmarks[LEFT_HEEL].y
    rh_y = landmarks[RIGHT_HEEL].y
    lh_x = landmarks[LEFT_HEEL].x
    rh_x = landmarks[RIGHT_HEEL].x
    lb_x = landmarks[LEFT_BIG_TOE].x
    rb_x = landmarks[RIGHT_BIG_TOE].x

    if lh_y < rh_y:
        front_heel_x = lh_x
        back_toe_x = rb_x
        if abs(abs(lh_x - lb_x) - abs(rh_x - rb_x)) > STABILITY_THRESHOLD:
            return False
    else:
        front_heel_x = rh_x
        back_toe_x = lb_x
        if abs(abs(rh_x - rb_x) - abs(lh_x - lb_x)) > STABILITY_THRESHOLD:
            return False

    if abs(front_heel_x - back_toe_x) < STABILITY_THRESHOLD:
        return True
    return False

def check_full_tandem(landmarks):
    """Checks for full tandem alignment."""
    lh_y = landmarks[LEFT_HEEL].y
    rh_y = landmarks[RIGHT_HEEL].y

    if lh_y < rh_y:
        front_heel_x = landmarks[LEFT_HEEL].x
        back_toe_x = landmarks[RIGHT_BIG_TOE].x
        front_toe_x = landmarks[LEFT_BIG_TOE].x
        back_heel_x = landmarks[RIGHT_HEEL].x
    else:
        front_heel_x = landmarks[RIGHT_HEEL].x
        back_toe_x = landmarks[LEFT_BIG_TOE].x
        front_toe_x = landmarks[RIGHT_BIG_TOE].x
        back_heel_x = landmarks[LEFT_HEEL].x

    if abs(front_heel_x - back_toe_x) < STABILITY_THRESHOLD and abs(front_toe_x - back_heel_x) < STABILITY_THRESHOLD:
        return True
    return False

def plot_hip_stability(task_idx, hip_x_data, hip_y_data, time_data, l_shoulder_x, r_shoulder_x, l_shoulder_y, r_shoulder_y):
    """Generates a detailed plot of hip and shoulder movement for a given task."""
    hip_center_x = np.array(hip_x_data)
    hip_center_y = np.array(hip_y_data)

    all_x = np.concatenate([hip_center_x, l_shoulder_x, r_shoulder_x]) if len(hip_center_x) else np.array([0.0])
    all_y = np.concatenate([hip_center_y, l_shoulder_y, r_shoulder_y]) if len(hip_center_y) else np.array([0.0])

    x_range = max(all_x.max() - all_x.min(), 0.05)
    y_range = max(all_y.max() - all_y.min(), 0.05)

    x_min, x_max = all_x.min() - 0.01, all_x.min() + x_range + 0.01
    y_min, y_max = all_y.min() - 0.01, all_y.min() + y_range + 0.01

    hip_x_std = np.std(hip_center_x) if hip_center_x.size > 0 else 0
    hip_y_std = np.std(hip_center_y) if hip_center_y.size > 0 else 0

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].plot(time_data, hip_center_x, label='Hip Center X', color='blue')
    axs[0].plot(time_data, l_shoulder_x, label='Left Shoulder X', linestyle='--', color='green')
    axs[0].plot(time_data, r_shoulder_x, label='Right Shoulder X', linestyle=':', color='red')
    axs[0].set_title('Lateral (X) Movement vs. Time')
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Normalized X Coordinate")
    axs[0].set_ylim(x_min, x_max)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time_data, hip_center_y, label='Hip Center Y', color='blue')
    axs[1].plot(time_data, l_shoulder_y, label='Left Shoulder Y', linestyle='--', color='green')
    axs[1].plot(time_data, r_shoulder_y, label='Right Shoulder Y', linestyle=':', color='red')
    axs[1].set_title('Vertical/Anterior-Posterior (Y) Movement vs. Time')
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Normalized Y Coordinate")
    axs[1].set_ylim(y_min, y_max)
    axs[1].legend()
    axs[1].grid(True)

    axs[2].scatter(hip_center_x, hip_center_y, c=time_data, cmap='viridis', label='Hip Path')
    axs[2].set_title('Hip Center Sway Path (X vs Y)')
    axs[2].set_xlabel("Normalized X Coordinate")
    axs[2].set_ylabel("Normalized Y Coordinate")
    axs[2].set_xlim(x_min, x_max)
    axs[2].set_ylim(y_min, y_max)
    axs[2].text(0.05, 0.95, f'Hip StdDev X: {hip_x_std:.4f}\nHip StdDev Y: {hip_y_std:.4f}',
                transform=axs[2].transAxes, fontsize=10, verticalalignment='top')
    axs[2].grid(True)

    fig.suptitle(f"MPPT Stability Analysis for: '{TASKS[task_idx]}'", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f"hip_shoulder_stability_{TASKS[task_idx].lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Graph saved as '{filename}'")

    plt.show()

def analyze_task_video(video_filename, task_idx):
    """Analyzes a single video for a specific task."""
    print(f"\n--- Analyzing {task_idx + 1}/{len(TASKS)}: {TASKS[task_idx]} ---")
    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_filename}'.")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Warning: Could not retrieve video FPS. Assuming 20 FPS (from recording).")
        fps = 20.0

    was_stable = False
    displacement_start_time = 0.0

    stable_time = 0.0
    displacement_time = 0.0
    displacement_events = []

    hip_x_data, hip_y_data, time_data = [], [], []
    l_shoulder_x, r_shoulder_x, l_shoulder_y, r_shoulder_y = [], [], [], []

    dt = 1 / fps

    with mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        while cap.isOpened():
            success, image = cap.read()

            if not success or (frame_idx / fps) > TASK_DURATION:
                break

            elapsed_time = frame_idx / fps

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            is_stable = False

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # ---- analysis task switch (unchanged) ----
                if task_idx == 0:
                    is_stable = check_feet_together(landmarks)
                elif task_idx == 1:
                    is_stable = check_semi_tandem(landmarks)
                elif task_idx == 2:
                    is_stable = check_full_tandem(landmarks)

                if is_stable:
                    stable_time += dt
                    if not was_stable:
                        displacement_end_time = elapsed_time
                        duration = displacement_end_time - displacement_start_time
                        if duration > 0.01:
                            displacement_events.append((displacement_start_time, duration))
                else:
                    displacement_time += dt
                    if was_stable:
                        displacement_start_time = elapsed_time

                was_stable = is_stable

                hip_center_x = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 2
                hip_center_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2

                hip_x_data.append(hip_center_x)
                hip_y_data.append(hip_center_y)
                time_data.append(elapsed_time)

                l_shoulder_x.append(landmarks[LEFT_SHOULDER].x)
                r_shoulder_x.append(landmarks[RIGHT_SHOULDER].x)
                l_shoulder_y.append(landmarks[LEFT_SHOULDER].y)
                r_shoulder_y.append(landmarks[RIGHT_SHOULDER].y)

                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

            current_task_time = min(elapsed_time, TASK_DURATION)

            total_current_time = stable_time + displacement_time
            if total_current_time > TASK_DURATION:
                stable_time -= (total_current_time - TASK_DURATION)
                stable_time = max(0, stable_time)
                displacement_time = min(TASK_DURATION, total_current_time) - stable_time

            display_stable = min(stable_time, TASK_DURATION)
            display_displaced = min(displacement_time, TASK_DURATION - display_stable)

            cv2.putText(image, f"ANALYZING: {TASKS[task_idx]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, f"Time: {current_task_time:.1f}s / {TASK_DURATION}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, f"Stable: {display_stable:.1f}s", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Displaced: {display_displaced:.1f}s", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(f'Analysis: {TASKS[task_idx]}', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n--- Results for {TASKS[task_idx]} ---")
    print(f"Total time in correct position: {stable_time:.2f}s")
    print(f"Total displaced time: {displacement_time:.2f}s")
    print("Displacement events:")
    if displacement_events:
        for start, duration in displacement_events:
            print(f"- Start: {start:.2f}s, Duration: {duration:.2f}s")
    else:
        print("- No displacements recorded.")

    plot_hip_stability(task_idx, hip_x_data, hip_y_data, time_data,
                       np.array(l_shoulder_x), np.array(r_shoulder_x), np.array(l_shoulder_y), np.array(r_shoulder_y))

    return stable_time, TASKS[task_idx]

# --- Main Program Execution ---
def main():
    # 1) Record three videos (10s each) with 2s buffers between segments
    print("\n\n*** Recording Three Segments (10s each) with 2s buffers ***")
    video_filenames = record_three_videos()

    # 2) Analyze the just-recorded videos
    stable_times = {}
    print("\n\n*** Starting Analysis of Recorded Videos ***")
    for i, video_filename in enumerate(video_filenames):
        if os.path.exists(video_filename):
            stable_time, task_name = analyze_task_video(video_filename, i)
            if stable_time is not None:
                stable_times[task_name] = stable_time
        else:
            print(f"Error: Video file '{video_filename}' not found. Skipping analysis.")
            stable_times[TASKS[i]] = 0.0

    # 3) Results & Final Score (UPDATED rubric only)
    print("\n*** Overall Results & Final Score (Updated Rubric) ***")
    feet_together_time = stable_times.get("Feet Together", 0.0)
    semi_tandem_time = stable_times.get("Semi-Tandem", 0.0)
    full_tandem_time = stable_times.get("Full Tandem", 0.0)

    print(f"Feet Together Stable Time: {feet_together_time:.2f}s")
    print(f"Semi-Tandem Stable Time: {semi_tandem_time:.2f}s")
    print(f"Full Tandem Stable Time: {full_tandem_time:.2f}s")

    final_score = calculate_score(feet_together_time, semi_tandem_time, full_tandem_time)
    print("\nScoring Table Applied:")
    print("4 → Full: 10s, Semi: 10s, Feet: 10s")
    print("3 → Full: 3–9s, Semi: 10s, Feet: 10s")
    print("2 → Full: 0–2s, Semi: 10s, Feet: 10s")
    print("1 → Full: 0s,   Semi: 0–9s, Feet: 10s")
    print("0 → Full: 0s,   Semi: 0–9s, Feet: 0–9s")
    print(f"\nFinal MPPT Score: {final_score}")
    print("*** MPPT Analysis Complete ***")

if __name__ == "__main__":
    main()
