import cv2
import os

video_path = r"C:\Users\talel\OneDrive\Documents\GitHub\portfolio\portfolio\Computer_vision\data_manipulator\Data_sample_Time_processing_&_Emotion_Detection\sample_cam1.mp4"
output_folder = r"C:\Users\talel\OneDrive\Documents\GitHub\portfolio\portfolio\Computer_vision\data_manipulator\data"
os.makedirs(output_folder, exist_ok=True)

# Find the highest existing frame number to avoid overwriting
existing_frames = [f for f in os.listdir(output_folder) if f.startswith("frame_") and f.endswith(".jpg")]
if existing_frames:
    existing_numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_frames]
    start_frame_number = max(existing_numbers) + 20  # Continue from next interval
else:
    start_frame_number = 0

cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0
save_interval = 20

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save every N-th frame
    if frame_count % save_interval == 0:
        output_frame_number = start_frame_number + frame_count
        frame_filename = os.path.join(output_folder, f"frame_{output_frame_number:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Processed {frame_count} total frames")
print(f"Saved {saved_count} frames (1 every {save_interval} frames)")
print(f"Starting from frame number: {start_frame_number}")
print(f"Output folder: {output_folder}")
