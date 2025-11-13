#!/usr/bin/env python3
"""
Real-time emoji display based on camera pose and facial expression detection.
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Configuration
SMILE_THRESHOLD = 0.05
SAD_THRESHOLD = 0.038
FIST_THRESHOLD = 0.12
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Load emoji images
try:
    smiling_emoji = cv2.imread("smile.jpg")
    straight_face_emoji = cv2.imread("plain.png")
    hands_up_emoji = cv2.imread("air.jpg")
    sad_emoji = cv2.imread("sad.jpg")
    thumbs_up_emoji = cv2.imread("thumb.jpg")

    if smiling_emoji is None:
        raise FileNotFoundError("smile.jpg not found")
    if straight_face_emoji is None:
        raise FileNotFoundError("plain.png not found")
    if hands_up_emoji is None:
        raise FileNotFoundError("air.jpg not found")
    if sad_emoji is None:
        raise FileNotFoundError("sad.jpg not found")
    if thumbs_up_emoji is None:
        raise FileNotFoundError("thumb.jpg not found")

    # Resize emojis
    smiling_emoji = cv2.resize(smiling_emoji, EMOJI_WINDOW_SIZE)
    straight_face_emoji = cv2.resize(straight_face_emoji, EMOJI_WINDOW_SIZE)
    hands_up_emoji = cv2.resize(hands_up_emoji, EMOJI_WINDOW_SIZE)
    thumbs_up_emoji = cv2.resize(thumbs_up_emoji, EMOJI_WINDOW_SIZE)
    
except Exception as e:
    print("Error loading emoji images!")
    print(f"Details: {e}")
    print("\nExpected files:")
    print("- smile.jpg (smiling face)")
    print("- plain.png (straight face)")
    print("- air.jpg (hands up)")
    exit()

blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[0], EMOJI_WINDOW_SIZE[1], 3), dtype=np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Emoji Output', WINDOW_WIDTH + 150, 100)

print("Controls:")
print("  Press 'q' to quit")
print("  Raise hands above shoulders for hands up")
print("  Smile for smiling emoji")
print("  Straight face for neutral emoji")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        current_state = "STRAIGHT_FACE"

        # Check for thumbs up
        left_thumb_up = False
        right_thumb_up = False
        results_pose = pose.process(image_rgb)

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark

            # Left Hand Landmarks
            left_thumb = landmarks[mp_pose.PoseLandmark.LEFT_THUMB]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            left_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
            left_pinky = landmarks[mp_pose.PoseLandmark.LEFT_PINKY]

            # Right Hand Landmarks
            right_thumb = landmarks[mp_pose.PoseLandmark.RIGHT_THUMB]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]
            right_pinky = landmarks[mp_pose.PoseLandmark.RIGHT_PINKY]

            # --- Thumbs Up Check (Revised Logic: Thumb Up + Index AND Pinky Close to Wrist) ---

            # Left Hand Distances (Euclidean distance between Index/Pinky tip and Wrist)
            left_index_dist = np.sqrt(
                (left_index.x - left_wrist.x)**2 + (left_index.y - left_wrist.y)**2
            )
            left_pinky_dist = np.sqrt(
                (left_pinky.x - left_wrist.x)**2 + (left_pinky.y - left_wrist.y)**2
            )
            # Right Hand Distances
            right_index_dist = np.sqrt(
                (right_index.x - right_wrist.x)**2 + (right_index.y - right_wrist.y)**2
            )
            right_pinky_dist = np.sqrt(
                (right_pinky.x - right_wrist.x)**2 + (right_pinky.y - right_wrist.y)**2
            )

            # Logic for Thumbs Up: Thumb is high AND Index/Pinky are close to the wrist (fist)
            left_thumb_is_up = left_thumb.y < left_wrist.y - 0.05
            left_is_fist = (left_index_dist < FIST_THRESHOLD) and (left_pinky_dist < FIST_THRESHOLD)
            left_thumbs_up = left_thumb_is_up and left_is_fist

            right_thumb_is_up = right_thumb.y < right_wrist.y - 0.05
            right_is_fist = (right_index_dist < FIST_THRESHOLD) and (right_pinky_dist < FIST_THRESHOLD)
            right_thumbs_up = right_thumb_is_up and right_is_fist

            if left_thumb_up or right_thumb_up:
                current_state = "THUMBS_UP"


        # Check for hands up if thumbs not up
        if current_state != "THUMBS_UP":
            results_pose = pose.process(image_rgb)
            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark
                
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                if (left_wrist.y < left_shoulder.y) or (right_wrist.y < right_shoulder.y):
                    current_state = "HANDS_UP"
        
        # Check facial expression if hands not up
        if current_state != "HANDS_UP" and current_state != "THUMBS_UP":
            results_face = face_mesh.process(image_rgb)
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    left_corner = face_landmarks.landmark[291]
                    right_corner = face_landmarks.landmark[61]
                    upper_lip = face_landmarks.landmark[13]
                    lower_lip = face_landmarks.landmark[14]

                    mouth_width = ((right_corner.x - left_corner.x)**2 + (right_corner.y - left_corner.y)**2)**0.5
                    mouth_height = ((lower_lip.x - upper_lip.x)**2 + (lower_lip.y - upper_lip.y)**2)**0.5
                    
                    if mouth_width > 0:
                        mouth_aspect_ratio = mouth_height / mouth_width
                        if mouth_aspect_ratio > SMILE_THRESHOLD:
                            current_state = "SMILING"
                        elif mouth_aspect_ratio < SAD_THRESHOLD:
                            current_state = "SAD"
                        else:
                            current_state = "STRAIGHT_FACE"
        
        # Select emoji based on state
        if current_state == "SMILING":
            emoji_to_display = smiling_emoji
            emoji_name = "😊"
        elif current_state == "STRAIGHT_FACE":
            emoji_to_display = straight_face_emoji
            emoji_name = "😐"
        elif current_state == "HANDS_UP":
            emoji_to_display = hands_up_emoji
            emoji_name = "🙌"
        elif current_state == "SAD":
            emoji_to_display = sad_emoji
            emoji_name = "😢"
        elif current_state == "THUMBS_UP":
            emoji_to_display = thumbs_up_emoji
            emoji_name = "👍"
        else:
            emoji_to_display = blank_emoji
            emoji_name = "❓"

        camera_frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        cv2.putText(camera_frame_resized, f'STATE: {current_state} {emoji_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(camera_frame_resized, 'Press "q" to quit', (10, WINDOW_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Camera Feed', camera_frame_resized)
        cv2.imshow('Emoji Output', emoji_to_display)

        print(f"Mouth ratio: {mouth_aspect_ratio:.3f}")


        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
