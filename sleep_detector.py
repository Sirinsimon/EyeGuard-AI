"""
Real-time Sleep Detection System using MediaPipe and OpenCV
Detects drowsiness by monitoring Eye Aspect Ratio (EAR)
Compatible with MediaPipe 0.10.8+
"""

import cv2
import numpy as np
import time
from scipy.spatial import distance
import pygame
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Eye landmark indices for MediaPipe Face Mesh (468 landmarks)
# Left eye indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye indices  
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# EAR threshold and time threshold
EAR_THRESHOLD = 0.25
SLEEP_TIME_THRESHOLD = 3.0  # seconds


def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) for given eye landmarks.
    
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    where p1-p6 are the eye landmark points
    
    Args:
        eye_landmarks: Array of 6 (x,y) coordinates for eye landmarks
        
    Returns:
        float: Eye Aspect Ratio value
    """
    # Compute euclidean distances between vertical eye landmarks
    vertical_1 = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    vertical_2 = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    
    # Compute euclidean distance between horizontal eye landmarks
    horizontal = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    
    # Calculate EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    
    return ear


def extract_eye_landmarks(face_landmarks, eye_indices, frame_width, frame_height):
    """
    Extract eye landmark coordinates from face mesh.
    
    Args:
        face_landmarks: MediaPipe face landmarks list
        eye_indices: List of landmark indices for the eye
        frame_width: Width of the video frame
        frame_height: Height of the video frame
        
    Returns:
        numpy array: Array of (x,y) coordinates for eye landmarks
    """
    landmarks = []
    for idx in eye_indices:
        landmark = face_landmarks[idx]
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        landmarks.append([x, y])
    
    return np.array(landmarks)


def detect_sleep_status(ear, eyes_closed_start_time):
    """
    Determine if person is sleeping based on EAR and time.
    
    Args:
        ear: Current Eye Aspect Ratio
        eyes_closed_start_time: Timestamp when eyes first closed (None if open)
        
    Returns:
        tuple: (status, updated_start_time, is_sleeping)
    """
    current_time = time.time()
    
    if ear < EAR_THRESHOLD:
        # Eyes are closed
        if eyes_closed_start_time is None:
            # Eyes just closed
            eyes_closed_start_time = current_time
            return "AWAKE", eyes_closed_start_time, False
        else:
            # Check how long eyes have been closed
            closed_duration = current_time - eyes_closed_start_time
            if closed_duration >= SLEEP_TIME_THRESHOLD:
                return "SLEEPING", eyes_closed_start_time, True
            else:
                return "AWAKE", eyes_closed_start_time, False
    else:
        # Eyes are open
        return "AWAKE", None, False


def draw_eye_landmarks(frame, left_eye, right_eye):
    """
    Draw eye landmarks on the frame.
    
    Args:
        frame: Video frame
        left_eye: Left eye landmark coordinates
        right_eye: Right eye landmark coordinates
    """
    # Draw left eye
    for point in left_eye:
        cv2.circle(frame, tuple(point), 2, (0, 255, 255), -1)
    cv2.polylines(frame, [left_eye], True, (0, 255, 255), 1)
    
    # Draw right eye
    for point in right_eye:
        cv2.circle(frame, tuple(point), 2, (0, 255, 255), -1)
    cv2.polylines(frame, [right_eye], True, (0, 255, 255), 1)


def draw_face_landmarks(frame, face_landmarks, frame_width, frame_height):
    """
    Draw face mesh landmarks on the frame.
    
    Args:
        frame: Video frame
        face_landmarks: List of face landmarks
        frame_width: Width of frame
        frame_height: Height of frame
    """
    # Draw a subset of landmarks for visualization (to avoid clutter)
    for i, landmark in enumerate(face_landmarks):
        if i % 5 == 0:  # Draw every 5th landmark to reduce clutter
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def play_alarm():
    """
    Play alarm sound when sleep is detected.
    """
    try:
        # Generate a simple beep sound
        sample_rate = 22050
        duration = 0.1
        frequency = 440
        t = np.linspace(0, duration, int(sample_rate * duration))
        wave = np.sin(2 * np.pi * frequency * t)
        sound = pygame.sndarray.make_sound((wave * 32767).astype(np.int16))
        sound.play()
    except Exception as e:
        # If sound fails, just pass
        pass


def main():
    """
    Main function to run the sleep detection system.
    """
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return
    
    # Initialize pygame mixer for alarm
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=1)
        alarm_enabled = True
    except Exception as e:
        print(f"Warning: Pygame not available, alarm disabled: {e}")
        alarm_enabled = False
    
    # Variables for tracking
    eyes_closed_start_time = None
    total_sleep_time = 0
    last_sleep_start = None
    alarm_playing = False
    last_alarm_time = 0
    
    print("Sleep Detection System Started")
    print("Press ESC to exit")
    print(f"MediaPipe version: {mp.__version__}")
    
    # Create FaceLandmarker
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    try:
        detector = vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Error: Could not create FaceLandmarker: {e}")
        print("\nPlease download the face_landmarker.task model file:")
        print("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
        print("\nSave it in the same directory as this script.")
        cap.release()
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect face landmarks
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        detection_result = detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                # Draw face mesh (subset of landmarks)
                draw_face_landmarks(frame, face_landmarks, frame_width, frame_height)
                
                # Extract eye landmarks
                left_eye = extract_eye_landmarks(
                    face_landmarks, LEFT_EYE, frame_width, frame_height
                )
                right_eye = extract_eye_landmarks(
                    face_landmarks, RIGHT_EYE, frame_width, frame_height
                )
                
                # Draw eye landmarks
                draw_eye_landmarks(frame, left_eye, right_eye)
                
                # Calculate EAR for both eyes
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Detect sleep status
                status, eyes_closed_start_time, is_sleeping = detect_sleep_status(
                    avg_ear, eyes_closed_start_time
                )
                
                # Handle sleep detection
                if is_sleeping:
                    color = (0, 0, 255)  # Red
                    if last_sleep_start is None:
                        last_sleep_start = time.time()
                    
                    # Play alarm (with cooldown to avoid continuous beeping)
                    current_time = time.time()
                    if alarm_enabled and (current_time - last_alarm_time) > 1.0:
                        play_alarm()
                        last_alarm_time = current_time
                else:
                    color = (0, 255, 0)  # Green
                    if last_sleep_start is not None:
                        total_sleep_time += time.time() - last_sleep_start
                        last_sleep_start = None
                
                # Display status on frame
                cv2.putText(
                    frame, 
                    f"Status: {status}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    color, 
                    2
                )
                
                # Display EAR value
                cv2.putText(
                    frame, 
                    f"EAR: {avg_ear:.2f}", 
                    (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
                
                # Display total sleep time
                current_sleep = total_sleep_time
                if last_sleep_start is not None:
                    current_sleep += time.time() - last_sleep_start
                
                cv2.putText(
                    frame, 
                    f"Sleep Time: {current_sleep:.1f}s", 
                    (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
        else:
            # No face detected
            cv2.putText(
                frame, 
                "No Face Detected", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
            eyes_closed_start_time = None
        
        # Display frame
        cv2.imshow('Sleep Detection System', frame)
        
        # Check for ESC key (27)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print(f"\nSession Summary:")
    print(f"Total Sleep Time: {total_sleep_time:.2f} seconds")
    print(f"Total Frames Processed: {frame_count}")
    print("Sleep Detection System Stopped")


if __name__ == "__main__":
    main()
