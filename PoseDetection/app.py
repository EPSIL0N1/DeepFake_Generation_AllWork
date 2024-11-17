import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Video input path
video_path = "./videos/one_person_saying_NO.mp4"
cap = cv2.VideoCapture(video_path)

# Output list to store pose keypoints
pose_keypoints = []

# Initialize Pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the frame with MediaPipe
        results = pose.process(image)

        # Draw pose landmarks on the frame (optional, for visualization)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Extract keypoints
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.visibility])
            pose_keypoints.append(keypoints)

        # Show the processed frame (optional, for debugging)
        cv2.imshow('Pose Detection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save the keypoints to a file (optional)
np.save("pose_keypoints.npy", np.array(pose_keypoints))

print("Pose extraction complete. Keypoints saved as pose_keypoints.npy.")
