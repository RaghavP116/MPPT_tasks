import cv2
import mediapipe as mp
import time
import os
import math

# --- Configuration and Constants ---
# Set this to the name of your existing video file in the same directory.
VIDEO_FILENAME = 'flexibility_test_recording.avi' 
# VIDEO_DURATION_SECONDS = 6 # No longer needed for recording
# FPS = 30.0 # No longer needed for recording

# MediaPipe Pose setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Landmark indices
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27 # Used as proxy for foot Y-coordinate
RIGHT_ANKLE = 28 # Used as proxy for foot Y-coordinate

# Thresholds (Normalized coordinates, 0.0 at top, 1.0 at bottom)
HAND_FOOT_Y_PROXIMITY_THRESHOLD = 0.01 # Max vertical distance (normalized) between wrist/ankle for Bent/Reach completion.

# Angle Thresholds
STAND_UP_HIP_ANGLE_MIN = 160.0 # Minimum angle (in degrees) for the hip joint (Knee-Hip-Shoulder) to confirm standing up.

# Scoring Metrics (Used for documentation, logic is implemented via if/elif)
SCORE_METRICS = {
    "0 to 2 seconds": 4,
    "2 to 4 seconds": 3,
    "4 to 6 seconds": 2,
    "> 6 seconds": 1,
    
}

def calculate_angle(a, b, c):
    """
    Calculates the angle (in degrees) between three points (landmarks).
    Used for both knee angle (Hip-Knee-Ankle) and hip angle (Knee-Hip-Shoulder).
    """
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]

    # Calculate angle using vector math
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = math.degrees(abs(radians))

    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# The record_video function has been removed as requested.

def analyze_video():
    """
    Analyzes an existing video file to determine flexibility score based on T_start to T_stand_up (T_END).
    The file name is defined by the VIDEO_FILENAME constant.
    """
    if not os.path.exists(VIDEO_FILENAME):
        print(f"Error: Video file '{VIDEO_FILENAME}' not found. Please ensure the video is in the same directory.")
        return

    print(f"\n--- Starting Video Analysis on {VIDEO_FILENAME} ---")

    cap = cv2.VideoCapture(VIDEO_FILENAME)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{VIDEO_FILENAME}'.")
        return

    # Initialize state variables
    initial_position_established = False
    initial_timestamp_sec = None # T_START (stable position)

    reach_completion_timestamp_sec = None # T_COMPLETED_REACH (Bent equivalent)
    stood_back_up_timestamp_sec = None # T_END

    is_reach_complete = False
    is_stood_back_up = False
    
    frame_count = 0

    # Initialize MediaPipe Pose with heavy model (model_complexity=2)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break 

            frame_count += 1
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Get frame dimensions for text placement
            height, width, _ = image.shape

            # --- Analysis Logic ---
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                try:
                    # Get Landmarks
                    l_wrist_y = landmarks[LEFT_WRIST].y
                    r_wrist_y = landmarks[RIGHT_WRIST].y
                    l_ankle_y = landmarks[LEFT_ANKLE].y
                    r_ankle_y = landmarks[RIGHT_ANKLE].y
                    
                    # Landmarks needed for hip angle check (Stand Up)
                    l_shoulder = landmarks[LEFT_SHOULDER]
                    l_hip = landmarks[LEFT_HIP]
                    l_knee = landmarks[LEFT_KNEE]
                    r_shoulder = landmarks[RIGHT_SHOULDER]
                    r_hip = landmarks[RIGHT_HIP]
                    r_knee = landmarks[RIGHT_KNEE]

                    # 1. Establish initial stable position (T_START)
                    if not initial_position_established and frame_count > 10:
                        initial_position_established = True
                        initial_timestamp_sec = timestamp_sec
                        print(f"!!! MEASUREMENT T_START !!! Video start time recorded at {initial_timestamp_sec:.2f}s (Start of measurement window)")
                        
                    if initial_position_established:
                        
                        # Phase 1: Detect Completed Reach/Bent (T_COMPLETED_REACH)
                        # Hand (Wrist Y) is near Foot (Ankle Y).
                        if not is_reach_complete:
                            
                            # Proximity Check (Vertical distance between hand and foot Y-coordinates)
                            is_left_near_foot = abs(l_wrist_y - l_ankle_y) < HAND_FOOT_Y_PROXIMITY_THRESHOLD
                            is_right_near_foot = abs(r_wrist_y - r_ankle_y) < HAND_FOOT_Y_PROXIMITY_THRESHOLD
                            
                            if is_left_near_foot or is_right_near_foot:
                                is_reach_complete = True
                                reach_completion_timestamp_sec = timestamp_sec
                                
                                # Print required information for T_COMPLETED_REACH
                                l_knee_angle = calculate_angle(l_hip, l_knee, landmarks[LEFT_ANKLE]) # Hip-Knee-Ankle
                                r_knee_angle = calculate_angle(r_hip, r_knee, landmarks[RIGHT_ANKLE]) # Hip-Knee-Ankle
                                
                                print("\n--- BENT/REACH CRITERIA MET (T_BENT) ---")
                                print(f"Bent Time: {reach_completion_timestamp_sec:.2f}s")
                                print(f"Knee Angles at Bent: Left: {l_knee_angle:.1f}° | Right: {r_knee_angle:.1f}°")
                                print("------------------------------------------")

                        # Phase 2: Detect Stand Back Up (T_END) - Requires reach to be complete
                        if is_reach_complete and not is_stood_back_up:
                            
                            # Calculate Hip Angle (Knee-Hip-Shoulder) for extension check
                            l_hip_angle = calculate_angle(l_knee, l_hip, l_shoulder)
                            r_hip_angle = calculate_angle(r_knee, r_hip, r_shoulder)
                            
                            # Check if both hips are near full extension (160 to 180 degrees)
                            is_fully_extended = (l_hip_angle >= STAND_UP_HIP_ANGLE_MIN and r_hip_angle >= STAND_UP_HIP_ANGLE_MIN)
                            
                            if is_fully_extended:
                                is_stood_back_up = True
                                stood_back_up_timestamp_sec = timestamp_sec
                                print(f"!!! STOOD BACK UP DETECTED (T_END) !!! at {stood_back_up_timestamp_sec:.2f}s.")
                                break # Stop analysis immediately for accurate T_END

                        # Draw landmarks and connections
                        mp_drawing.draw_landmarks(
                            image,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2)
                        )

                except Exception as e:
                    # print(f"Error during landmark processing: {e}") # Optional: for debugging
                    pass
            
            # --- Display Results and Status (Moved to bottom of the frame) ---
            
            # Determine the current status text
            if not initial_position_established:
                status_text = "Status: WAITING FOR START"
                color = (255, 255, 255)
            elif not is_reach_complete:
                status_text = "Status: READY (Bent: Hand must reach foot level)"
                color = (0, 255, 0)
            elif not is_stood_back_up:
                status_text = "Status: REACHED (Stand Back Up Now)"
                color = (255, 165, 0) # Orange
            else:
                status_text = "Status: FINISHED"
                color = (0, 0, 255)

            # Time calculation for display
            display_time = 0.0
            if initial_timestamp_sec:
                if stood_back_up_timestamp_sec:
                    display_time = stood_back_up_timestamp_sec - initial_timestamp_sec
                else:
                    display_time = timestamp_sec - initial_timestamp_sec

            # Text Placement (Bottom of the window)
            text_y_status = height - 50 
            text_y_time = height - 20
            
            cv2.putText(image, status_text, (20, text_y_status), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            time_text = f"Total Elapsed Time: {display_time:.2f}s"
            cv2.putText(image, time_text, (20, text_y_time), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('Flexibility Test Analysis', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # --- Final Score Calculation ---
        cap.release()
        cv2.destroyAllWindows()

        if initial_timestamp_sec and is_reach_complete and is_stood_back_up:
            final_time = stood_back_up_timestamp_sec - initial_timestamp_sec
            final_score = 0
            
            # Apply new scoring metrics based on time (T_START to T_END)
            if final_time <= 2.0:
                final_score = 4
            elif final_time <= 4.0:
                final_score = 3
            elif final_time <= 6.0:
                final_score = 2
            else: # final_time > 6.0
                 final_score = 1 
            
            result_text = f"Total Time (T_START to T_END): {final_time:.2f}s. Final Score: {final_score}"

            print("\n--- FINAL SCORING RESULTS ---")
            print(f"Time T_START (Video Start): {initial_timestamp_sec:.2f}s")
            print(f"Time T_REACH (Bent/Reach Completion): {reach_completion_timestamp_sec:.2f}s")
            print(f"Time T_END (Stood Back Up): {stood_back_up_timestamp_sec:.2f}s")
            print(result_text)
            print("-----------------------------")
            
        else:
            final_score = 0 # Explicitly set score to 0 for failure/incomplete task
            print("\n--- ANALYSIS FAILED (Score 0) ---")
            
            if not initial_position_established:
                print("Failure Reason: Could not detect a stable starting position.")
                print(f"Time Taken: N/A. Final Score: {final_score}")
            elif not is_reach_complete:
                # --- MODIFIED BLOCK FOR NEW REQUIREMENT ---
                # This condition means the initial position was established, but the reach criteria was never met.
                print("Failure Reason: Bent/Reach Completion was not achieved (Hand did not reach foot level).")
                print("*** The person was unable to touch the ground (or reach the foot Y-level) throughout the video. ***") # <-- Added message
                print(f"Time Taken: N/A. Final Score: {final_score}")
                print("Status: The person didn't bend or meet the criteria.")
                # --- END MODIFIED BLOCK ---
            elif not is_stood_back_up:
                print(f"Failure Reason: The person did not stand back up (Hip Angle < {STAND_UP_HIP_ANGLE_MIN}° required) before the video ended.")
                print(f"Time Taken: N/A. Final Score: {final_score}")
                print("Status: The person didn't complete the stand-up phase.")
            
            print("---------------------------------")


if __name__ == '__main__':
    # Directly analyze the existing video file
    analyze_video()