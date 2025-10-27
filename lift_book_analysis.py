# analyze_fist_up_down_from_video.py
import os
import cv2
import mediapipe as mp
import time
import math
import argparse

# --- Configuration ---
VIDEO_INPUT_FILENAME  = "input_video.mp4"     # default existing video
VIDEO_OUTPUT_FILENAME = "output_analysis.mp4"

# --- Fist Detection Heuristic Parameters ---
THUMB_PROXIMITY_THRESHOLD = 0.08  # normalized distance
MIN_PROXIMITY_CHECKS = 2

# --- State Definitions for Sequential Logic ---
STATE_WAIT_FOR_FIST                 = 0
STATE_FIST_CONFIRMED_WAIT_FOR_UP    = 1
STATE_HAND_UP_CONFIRMED_WAIT_FOR_DOWN = 2
STATE_SEQUENCE_COMPLETE             = 3

# Helper: Euclidean distance between two MediaPipe landmarks
def calculate_distance(lm1, lm2):
    return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)

def analyze_video_sequential_checks(video_path):
    """
    Analyze a pre-recorded video for:
      1) Thumb tip close to index & middle tips (trigger),
      2) Then hand above shoulder,
      3) Then hand below shoulder (stop timer).
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path} for analysis.")
        return

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0

    out_analysis = cv2.VideoWriter(
        VIDEO_OUTPUT_FILENAME,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    if not out_analysis.isOpened():
        print("Error: Could not open video writer for analysis output.")
        cap.release()
        return

    print(f"Analyzing '{video_path}' …")

    current_state = STATE_WAIT_FOR_FIST
    start_event_time = 0.0
    sequence_end_time = 0.0
    initial_fist_detected_ever = False

    display_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # If sequence completed, write last frame and stop
        if current_state == STATE_SEQUENCE_COMPLETE:
            if display_frame is not None:
                out_analysis.write(display_frame)
            break

        current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        display_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Pose: shoulders ---
        pose_results = pose.process(rgb_frame)
        shoulder_y_reference = None

        if pose_results.pose_landmarks:
            left_shoulder  = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Visualize shoulders (optional)
            def norm_to_px(norm, W, H):
                x = int(norm.x * W); y = int(norm.y * H)
                return x, y

            ls_px = norm_to_px(left_shoulder, frame_width, frame_height)
            rs_px = norm_to_px(right_shoulder, frame_width, frame_height)
            cv2.circle(display_frame, ls_px, 8, (0, 255, 255), -1)
            cv2.circle(display_frame, rs_px, 8, (0, 255, 255), -1)

            shoulder_y_left  = left_shoulder.y  * frame_height
            shoulder_y_right = right_shoulder.y * frame_height
            shoulder_y_reference = min(shoulder_y_left, shoulder_y_right)  # top-most shoulder

            # Reference line
            cv2.line(display_frame, (0, int(shoulder_y_reference)),
                     (frame_width, int(shoulder_y_reference)), (255, 128, 0), 2)
            cv2.putText(display_frame, "Shoulder Ref",
                        (frame_width - 180, int(shoulder_y_reference) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

        # --- Hands & trigger ---
        hands_results = hands.process(rgb_frame)

        is_fist_closed_in_frame = False
        hand_wrist_y = -1

        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

                if current_state == STATE_WAIT_FOR_FIST:
                    thumb_tip  = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip  = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                    proximity_checks = 0
                    if calculate_distance(thumb_tip, index_tip)  < THUMB_PROXIMITY_THRESHOLD:  proximity_checks += 1
                    if calculate_distance(thumb_tip, middle_tip) < THUMB_PROXIMITY_THRESHOLD:  proximity_checks += 1

                    if proximity_checks >= MIN_PROXIMITY_CHECKS:
                        is_fist_closed_in_frame = True
                        initial_fist_detected_ever = True

                hand_wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame_height)
                break  # single-hand logic for this sequence

        # --- State machine ---
        status_text = "N/A"
        temporary_message = ""

        if current_state == STATE_WAIT_FOR_FIST:
            status_text = "1. Waiting for Thumb-to-Fingertips Proximity…"
            if is_fist_closed_in_frame:
                current_state = STATE_FIST_CONFIRMED_WAIT_FOR_UP
                start_event_time = current_video_time
                print(f"[{current_video_time:.2f}s] Check 1: Thumb proximity detected. Timer started.")
                temporary_message = "✅ Trigger Activated! Timer Started!"

        elif current_state == STATE_FIST_CONFIRMED_WAIT_FOR_UP:
            status_text = "2. Trigger Confirmed. Waiting for Hand UP (Above Shoulder)…"
            if shoulder_y_reference is not None and hand_wrist_y != -1:
                if hand_wrist_y < shoulder_y_reference:  # above
                    current_state = STATE_HAND_UP_CONFIRMED_WAIT_FOR_DOWN
                    print(f"[{current_video_time:.2f}s] Check 2: Hand is UP (first time).")

        elif current_state == STATE_HAND_UP_CONFIRMED_WAIT_FOR_DOWN:
            status_text = "3. Hand UP. Waiting for Hand DOWN (Below Shoulder)…"
            if shoulder_y_reference is not None and hand_wrist_y != -1:
                if hand_wrist_y >= shoulder_y_reference:  # down
                    current_state = STATE_SEQUENCE_COMPLETE
                    sequence_end_time = current_video_time
                    print(f"[{current_video_time:.2f}s] Check 3: Hand is DOWN. Sequence COMPLETE! Timer Stopped.")

        elif current_state == STATE_SEQUENCE_COMPLETE:
            status_text = f"Sequence COMPLETE! Time: {sequence_end_time - start_event_time:.2f}s"

        # --- HUD ---
        cv2.putText(display_frame, f"Current State: {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        if temporary_message:
            cv2.putText(display_frame, temporary_message, (10, frame_height - 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0,255,255), 3, cv2.LINE_AA)

        if current_state == STATE_WAIT_FOR_FIST:
            cv2.putText(display_frame, f"Trigger: {'ACTIVATED' if is_fist_closed_in_frame else 'INACTIVE'}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0) if is_fist_closed_in_frame else (0,0,255), 2, cv2.LINE_AA)
        elif current_state == STATE_FIST_CONFIRMED_WAIT_FOR_UP:
            cv2.putText(display_frame, "Trigger: (Initial Activated, not checked)", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2, cv2.LINE_AA)
        else:
            cv2.putText(display_frame, "Trigger: (Not Checked)", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,128,128), 2, cv2.LINE_AA)

        if shoulder_y_reference is not None and hand_wrist_y != -1:
            if current_state in (STATE_FIST_CONFIRMED_WAIT_FOR_UP, STATE_HAND_UP_CONFIRMED_WAIT_FOR_DOWN):
                hand_is_above = hand_wrist_y < shoulder_y_reference
                cv2.putText(display_frame, f"Above Shoulder: {'YES' if hand_is_above else 'NO'}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0,255,0) if hand_is_above else (0,0,255), 2, cv2.LINE_AA)

        if start_event_time > 0 and current_state != STATE_SEQUENCE_COMPLETE:
            cv2.putText(display_frame, f"Elapsed: {current_video_time - start_event_time:.2f}s", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)

        # Write & show
        out_analysis.write(display_frame)
        cv2.imshow('Analysis Output', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Analysis stopped manually.")
            break

    cap.release()
    out_analysis.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()
    print(f"Analysis complete. Output video saved as '{VIDEO_OUTPUT_FILENAME}'")

    # Summary
    if current_state == STATE_SEQUENCE_COMPLETE:
        total_time_taken = sequence_end_time - start_event_time
        print("\n--- SEQUENCE SUCCESSFULLY COMPLETED ---")
        print(f"Total time (Trigger -> Up -> Down): {total_time_taken:.2f} s")
    elif current_state == STATE_WAIT_FOR_FIST and not initial_fist_detected_ever:
        print("\n--- SEQUENCE FAILED: Initial trigger (Thumb Proximity) never detected. ---")
    else:
        print("\n--- SEQUENCE NOT COMPLETED ---")
        print(f"Sequence ended in state: {current_state}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a prerecorded video for thumb-proximity trigger → hand up → hand down.")
    parser.add_argument("--video", "-v", default=VIDEO_INPUT_FILENAME, help="Path to existing video file")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")

    analyze_video_sequential_checks(args.video)
