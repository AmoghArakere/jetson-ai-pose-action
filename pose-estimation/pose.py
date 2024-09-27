import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to process frame
def process_frame(frame):
    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect poses
    results = pose.process(image)
    
    # Draw pose landmarks on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    
    return frame

# Open the video file
video = cv2.VideoCapture('dance.mp4')

# Get video properties
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object
out = cv2.VideoWriter('output_dance_pose.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Process the video
frame_count = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    # Process the frame
    processed_frame = process_frame(frame)
    
    # Write the frame to the output video
    out.write(processed_frame)
    
    frame_count += 1
    if frame_count % 30 == 0:  # Print progress every 30 frames
        print(f"Processed {frame_count} frames")

# Release the video capture and writer objects
video.release()
out.release()

print("Video processing complete. Output saved as 'output_dance_pose.mp4'")

# Display the first frame of the processed video
first_frame = cv2.VideoCapture('output_dance_pose.mp4')
ret, frame = first_frame.read()
if ret:
    from IPython.display import display, Image
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("First frame of processed dance video with pose estimation")
    plt.savefig('first_frame_dance_pose.jpg')
    display(Image('first_frame_dance_pose.jpg'))
    
first_frame.release()

print("To download the processed video, look for 'output_dance_pose.mp4' in the file browser on the left.")
