import cv2

# Path to the video file
video_path = "./videos/one_person_saying_NO.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Read the first frame
    ret, frame = cap.read()
    
    if ret:
        # Save the first frame as an image file
        output_image_path = 'first_frame.jpg'
        cv2.imwrite("./PoseDetection/firstFrame.jpeg", frame)
        print(f"First frame saved as '{output_image_path}'.")
    else:
        print("Error: Could not read the first frame.")

# Release the video capture object
cap.release()
