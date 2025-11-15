import cv2
import time
from ultralytics import YOLO
import os

class YOLOROITracker:
    def __init__(self, model_path, video_path=None):
        """
        Initialize YOLO ROI Tracker
        
        Args:
            model_path: Path to YOLO model
            video_path: Path to video file (None for webcam)
        """
        self.model = YOLO(model_path)
        self.video_path = video_path
        
        # Region of Interest (bottom 1/5th of frame)
        self.roi_height_ratio = 0.2  # Bottom 1/5th = 20%
        
        # Tracking variables
        self.person_in_roi = False
        self.roi_start_time = None
        self.total_time_in_roi = 0.0
        self.current_session_time = 0.0
        
        # Statistics
        self.roi_entries = 0
        self.frame_count = 0
        
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            box1: (x1, y1, x2, y2) coordinates
            box2: (x1, y1, x2, y2) coordinates
        
        Returns:
            IoU value (0 to 1) and intersection area
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        # Calculate intersection area
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0, 0
        
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou, intersection_area
    
    def process_frame(self, frame):
        """Process a single frame"""
        self.frame_count += 1
        height, width = frame.shape[:2]
        
        # Define ROI (bottom 1/5th of frame)
        roi_y1 = int(height * (1 - self.roi_height_ratio))
        roi_box = (0, roi_y1, width, height)
        
        # Draw ROI rectangle (static)
        cv2.rectangle(frame, (0, roi_y1), (width, height), (0, 255, 255), 3)
        cv2.putText(frame, "Region of Interest", (10, roi_y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        # Track if any person is in ROI this frame
        person_detected_in_roi = False
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Filter for person class (class 0 in COCO dataset)
                if cls == 0 and conf > 0.5:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_box = (x1, y1, x2, y2)
                    
                    # Calculate overlap with ROI
                    iou, intersection_area = self.calculate_iou(person_box, roi_box)
                    
                    # Determine color based on ROI overlap
                    if iou > 0 or intersection_area > 0:
                        color = (0, 255, 0)  # Green if in ROI
                        person_detected_in_roi = True
                        label = f"Person {conf:.2f} [IN ROI]"
                    else:
                        color = (255, 0, 0)  # Blue if outside ROI
                        label = f"Person {conf:.2f}"
                    
                    # Draw person bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Show overlap info
                    if iou > 0:
                        overlap_text = f"Overlap: {iou*100:.1f}%"
                        cv2.putText(frame, overlap_text, (x1, y2 + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update timing based on ROI presence
        current_time = time.time()
        
        if person_detected_in_roi:
            if not self.person_in_roi:
                # Person just entered ROI
                self.person_in_roi = True
                self.roi_start_time = current_time
                self.roi_entries += 1
                self.current_session_time = 0.0
            else:
                # Person still in ROI, update session time
                self.current_session_time = current_time - self.roi_start_time
        else:
            if self.person_in_roi:
                # Person just left ROI
                self.person_in_roi = False
                if self.roi_start_time:
                    session_duration = current_time - self.roi_start_time
                    self.total_time_in_roi += session_duration
                self.roi_start_time = None
                self.current_session_time = 0.0
        
        # Draw statistics
        self.draw_statistics(frame)
        
        return frame
    
    def draw_statistics(self, frame):
        """Draw statistics on frame"""
        y_offset = 30
        line_height = 30
        
        # Background for statistics
        cv2.rectangle(frame, (10, 10), (400, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 160), (255, 255, 255), 2)
        
        # Statistics text
        stats = [
            f"Frame: {self.frame_count}",
            f"ROI Entries: {self.roi_entries}",
            f"Total Time in ROI: {self.total_time_in_roi:.2f}s",
            f"Current Session: {self.current_session_time:.2f}s",
            f"Status: {'IN ROI' if self.person_in_roi else 'Outside ROI'}"
        ]
        
        for i, stat in enumerate(stats):
            color = (0, 255, 0) if self.person_in_roi and i == 4 else (255, 255, 255)
            cv2.putText(frame, stat, (20, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def run(self):
        """Main loop to process video or webcam"""
        # Open video source
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            print(f"Processing video: {self.video_path}")
        else:
            cap = cv2.VideoCapture(0)
            print("Opening webcam...")
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.video_path else -1
        
        print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")
        if total_frames > 0:
            print(f"Total frames: {total_frames}")
        
        # Processing loop
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video or camera disconnected")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display frame
            cv2.imshow('YOLO ROI Tracker - Press Q to quit', processed_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                print("\nQuitting...")
                break
        
        # Cleanup
        elapsed_time = time.time() - start_time
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        self.print_final_statistics(elapsed_time)
    
    def print_final_statistics(self, elapsed_time):
        """Print final statistics"""
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Processing time: {elapsed_time:.2f}s")
        print(f"Average FPS: {self.frame_count/elapsed_time:.2f}")
        print(f"ROI entries: {self.roi_entries}")
        print(f"Total time person in ROI: {self.total_time_in_roi:.2f}s")
        if self.roi_entries > 0:
            print(f"Average time per ROI session: {self.total_time_in_roi/self.roi_entries:.2f}s")
        print("="*50)


def main():
    # Configuration
    model_path = r"../models/yolo11s.pt"
    
    # Choose video source
    # Option 1: Use webcam (set to None)
    # video_path = None
    
    # Option 2: Use video file
    video_path = r"../data_manipulator/Data_sample_Time_processing_&_Emotion_Detection/sample_cam2.mp4"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Check if video exists (if using video file)
    if video_path and not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    
    # Initialize and run tracker
    tracker = YOLOROITracker(model_path, video_path)
    tracker.run()


if __name__ == "__main__":
    main()
