import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt

def analyze_existing_video(filename="blurred_output.mp4"):
    """
    Analyzes an existing video file, stops immediately after the first 360-degree turn,
    and generates a plot of the shoulder landmark movement.
    """
    print(f"Starting analysis of {filename}...")
    
    # --- Part 1: Analysis Setup ---
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print(f"Error: Could not open video file {filename}. Please ensure the file exists in the same directory.")
        return

    # Tracking variables
    turn_count = 0
    turn_details = []
    pause_times = []
    
    # Data for Plotting
    shoulder_x_data = []
    shoulder_z_data = []
    
    # Rotation variables
    accumulated_angle = 0
    previous_angle = None
    
    # Stability tracking variables
    hip_y_coords = []
    leg_y_coords = []
    
    # Pause detection variables
    is_pausing = False
    pause_start_time = None
    pause_threshold = 0.5
    
    # Video timing variables using OpenCV properties for accuracy
    turn_start_video_time = None
    # FLAG: Tracks if the significant turn has begun (i.e., angle >= 10 deg)
    turn_in_progress = False 
    
    # Variable to hold the actual end time of the analysis loop
    final_analysis_stop_time = 0.0
    
    # --- Part 2: Analysis Loop with Visualization ---
    while cap.isOpened():
        ret, frame = cap.read()
        
        # STOP CONDITION: Break if end of video OR 1 turn complete
        if not ret or turn_count >= 1:
            break

        # Get the current video time in seconds
        elapsed_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        frame = cv2.resize(frame, (800, 600))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # Draw landmarks and connections
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # --- Data Collection for Graph ---
            # Track average X (horizontal) and Z (depth) of shoulders
            avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            avg_shoulder_z = (left_shoulder.z + right_shoulder.z) / 2
            shoulder_x_data.append(avg_shoulder_x)
            shoulder_z_data.append(avg_shoulder_z)
            
            # --- Rotation Logic ---
            hip_vector = np.array([left_hip.x - right_hip.x, left_hip.y - right_hip.y, left_hip.z - right_hip.z])
            current_angle = np.degrees(np.arctan2(hip_vector[0], -hip_vector[2]))
            
            if previous_angle is None:
                previous_angle = current_angle
            
            delta_angle = current_angle - previous_angle
            if delta_angle > 180:
                delta_angle -= 360
            elif delta_angle < -180:
                delta_angle += 360
            
            # Pause detection (only run if turn is in progress)
            if turn_in_progress:
                angle_change_rate = abs(delta_angle)
                if angle_change_rate < 1:
                    if not is_pausing:
                        pause_start_time = elapsed_video_time
                        is_pausing = True
                elif is_pausing:
                    pause_duration = elapsed_video_time - pause_start_time
                    if pause_duration >= pause_threshold:
                        pause_times.append(pause_duration)
                        cv2.putText(frame, f'PAUSED: {pause_duration:.2f}s', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    is_pausing = False
            
            accumulated_angle += delta_angle
            
            # Start timer when angle >= 20 degrees
            if not turn_in_progress and abs(accumulated_angle) >= 10:
                turn_start_video_time = elapsed_video_time
                turn_in_progress = True
                print(f"Turn started at {turn_start_video_time:.2f}s (Angle >= 10 deg)")
            
            # Check for a full 360-degree turn
            if turn_in_progress and abs(accumulated_angle) >= 360: # Ensure the turn has officially started
                turn_end_video_time = elapsed_video_time
                turn_duration = turn_end_video_time - turn_start_video_time
                turn_details.append({
                    'turn': turn_count + 1,
                    'start_time': turn_start_video_time,
                    'end_time': turn_end_video_time,
                    'duration': turn_duration
                })
                turn_count += 1
                
                # STORE THE ACTUAL STOP TIME and PRINT it
                final_analysis_stop_time = turn_end_video_time
                print(f"Analysis Stopping Time (Turn End Time): {final_analysis_stop_time:.2f}s") 
                
                # STOP ANALYSIS AFTER 1 TURN
                print("\n--- FIRST 360-DEGREE TURN COMPLETE. STOPPING ANALYSIS. ---")
                cap.release()
                cv2.destroyAllWindows()
                break # Exit the while loop
                
            previous_angle = current_angle

            # --- Stability Tracking ---
            hip_y_coords.append((left_hip.y + right_hip.y) / 2)
            leg_y_coords.append((left_ankle.y + right_ankle.y) / 2)

        # --- Display Metrics on Screen ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Turns: {turn_count}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Time: {elapsed_video_time:.2f}s', (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Rotation: {abs(accumulated_angle):.2f} deg', (10, 110), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if not turn_in_progress:
             cv2.putText(frame, 'Awaiting 10 deg Start...', (10, 150), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('360 Turn Analysis (Stops after 1st turn)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # --- Part 3: Reporting the Final Results ---
    stability_hip_std = np.std(hip_y_coords) if hip_y_coords else 0
    stability_leg_std = np.std(leg_y_coords) if leg_y_coords else 0
    
    print("\n--- MPPT 360-Degree Turn Analysis Results ---")
    print(f"Total Turns Detected: {turn_count}")
    
    if turn_details:
        print("\nIndividual Turn Details:")
        for details in turn_details:
            print(f"  Turn {details['turn']}: Start={details['start_time']:.2f}s (10 deg mark), End={details['end_time']:.2f}s, Duration={details['duration']:.2f}s")
            
    print(f"\nStability (Hip Fluctuation - lower is better): {stability_hip_std:.4f}")
    print(f"Stability (Leg Fluctuation - lower is better): {stability_leg_std:.4f}")
    
    if pause_times:
        print("\nPauses Detected:")
        for i, t in enumerate(pause_times):
            print(f"  Pause {i+1}: {t:.2f} seconds")
    else:
        print("\nNo significant pauses detected.")
        
    # --- Part 4: Generate Plot ---
    if shoulder_x_data:
        # Use the stored final stop time for the plot's time axis end point
        final_time_for_plot = final_analysis_stop_time if final_analysis_stop_time > 0 else elapsed_video_time
        time_points = np.linspace(0, final_time_for_plot, len(shoulder_x_data))

        plt.figure(figsize=(10, 6))
        
        # Plot X-coordinate (horizontal movement)
        plt.plot(time_points, shoulder_x_data, label='Normalized X (Horizontal)', color='blue')
        
        # Plot Z-coordinate (depth movement/rotation proxy)
        plt.plot(time_points, shoulder_z_data, label='Normalized Z (Depth/Sway)', color='red', linestyle='--')
        
        plt.title('Shoulder Center Movement During First Turn')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Normalized Coordinate Value (0.0 to 1.0)')
        plt.legend()
        plt.grid(True)
        plt.show()

# --- Main execution flow ---
if __name__ == "__main__":
    # Ensure you have a video named "blurred_output.mp4" in the same directory
    analyze_existing_video(filename="blurred_output.mp4")