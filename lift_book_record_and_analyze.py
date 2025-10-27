# Import necessary libraries
import cv2
import mediapipe as mp
import time
import os

# --- Configuration and Initialization ---
VIDEO_FILE = "recorded_video.mp4"
RECORD_DURATION = 10  # seconds

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Video Recording Function ---
def record_video(filename, duration):
    """
    Records video from the webcam for a specified duration.
    
    Args:
        filename (str): The name of the output video file.
        duration (int): The recording duration in seconds.
    """
    print(f"Starting video recording for {duration} seconds...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    # Get video properties to set up the writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()
    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break

        # Display a "Recording..." message on the frame
        cv2.putText(frame, "Recording...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Recording Video', frame)

        # Write the frame to the output file
        out.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when the job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {filename}")
    return True

# --- Video Analysis Function ---
def analyze_video(filename):
    """
    Analyzes a video file to check for specific actions using MediaPipe.
    
    Args:
        filename (str): The path to the video file.
    """
    if not os.path.exists(filename):
        print(f"Error: Video file '{filename}' not found.")
        return

    print(f"Starting analysis of '{filename}'...")
    cap = cv2.VideoCapture(filename)
    
    # Flags to track the state of the actions
    is_holding_book = False
    is_standing_upright = False
    is_book_placed = False

    # A simple state tracker to manage the sequence of events
    current_state = "Waiting to pick up"

    # Use a variable to store the person's approximate hip height for relative calculations
    hip_height_ref = None

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video file. Exiting.")
                break

            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image with MediaPipe Pose
            results = pose.process(frame_rgb)

            # Draw the pose landmarks on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates for key landmarks
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                
                # Get the average hip and shoulder y-coordinates (normalized to frame height)
                avg_hip_y = (left_hip.y + right_hip.y) / 2
                avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                
                # Calculate hand distance and height
                hand_distance_x = abs(left_wrist.x - right_wrist.x)
                avg_hand_y = (left_wrist.y + right_wrist.y) / 2

                # --- Condition 1: Check if the person is holding the book ---
                # This is a simplified check. We assume "holding" means hands are low and close together.
                if current_state == "Waiting to pick up" and avg_hand_y > avg_hip_y and hand_distance_x < 0.15:
                    is_holding_book = True
                    current_state = "Lifting and holding"
                    print("Status: Person is holding the book.")

                # --- Condition 2: Check if the person is standing upright ---
                # We check for a stable posture where hips are below shoulders
                if current_state == "Lifting and holding" and avg_hip_y > avg_shoulder_y and (avg_hip_y - avg_shoulder_y) > 0.15:
                     is_standing_upright = True
                     current_state = "Standing upright"
                     print("Status: Person has stood up straight with the book.")
                     hip_height_ref = avg_hip_y

                # --- Condition 3: Check if the person placed the book on the table ---
                # We assume the table is at a height between the hips and hands when standing.
                # Check for hands moving down from the upright position and then separating.
                if current_state == "Standing upright":
                    # Define table height as a range relative to the hip height
                    table_height_min = hip_height_ref - 0.1
                    table_height_max = hip_height_ref + 0.1
                    
                    if avg_hand_y > table_height_min and avg_hand_y < table_height_max and hand_distance_x > 0.3:
                        is_book_placed = True
                        current_state = "Book placed on table"
                        print("Status: Book has been successfully placed on the table.")
            
            # --- Display analysis results on the frame ---
            
            # Display current state
            cv2.putText(frame, f"State: {current_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display individual checks
            color_holding = (0, 255, 0) if is_holding_book else (0, 0, 255)
            cv2.putText(frame, f"1. Holding Book: {is_holding_book}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_holding, 2)
            
            color_standing = (0, 255, 0) if is_standing_upright else (0, 0, 255)
            cv2.putText(frame, f"2. Standing Upright: {is_standing_upright}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_standing, 2)
            
            color_placed = (0, 255, 0) if is_book_placed else (0, 0, 255)
            cv2.putText(frame, f"3. Book Placed: {is_book_placed}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_placed, 2)

            # Display the resulting frame
            cv2.imshow('Video Analysis', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release everything when the job is finished
    cap.release()
    cv2.destroyAllWindows()

# --- Main execution block ---
if __name__ == "__main__":
    # Step 1: Record the video
    if record_video(VIDEO_FILE, RECORD_DURATION):
        # Step 2: Analyze the recorded video
        analyze_video(VIDEO_FILE)
