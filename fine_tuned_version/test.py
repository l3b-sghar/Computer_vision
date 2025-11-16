"""
Fine-tuned YOLO Model Inference
Detects: personFF (person facing forward), personFB (person facing backward), counter
Starts timing only when personFF and counter bounding boxes overlap
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
import os

class FineTunedYOLOTracker:
    def __init__(self, model_path, video_path=None):
        """
        Initialize Fine-tuned YOLO Tracker
        
        Args:
            model_path: Path to fine-tuned YOLO model (best.pt)
            video_path: Path to video file (None for webcam)
        """
        self.model = YOLO(model_path)
        self.video_path = video_path
        
        # Class names mapping
        self.class_names = {
            0: 'personFF',   # Person Facing Forward
            1: 'personFB',   # Person Facing Backward
            2: 'counter'     # Counter
        }
        
        # Tracking variables
        self.person_at_counter = False
        self.counter_start_time = None
        self.total_time_at_counter = 0.0
        self.current_session_time = 0.0
        
        # Statistics
        self.counter_entries = 0
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()
        
        # Colors for each class
        self.class_colors = {
            'personFF': (0, 255, 0),    # Green
            'personFB': (0, 165, 255),  # Orange
            'counter': (0, 255, 255)    # Yellow
        }
    
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
        """Process a single frame with fine-tuned YOLO model"""
        self.frame_count += 1
        height, width = frame.shape[:2]
        
        # Calculate FPS
        current_time = time.time()
        if current_time - self.fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.fps_time)
            self.fps_time = current_time
            self.frame_count = 0
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        # Store detected objects
        person_ff_boxes = []
        person_fb_boxes = []
        counter_boxes = []
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2, y2)
                
                # Get class name
                class_name = self.class_names.get(cls, 'unknown')
                
                # Filter by confidence threshold
                if conf > 0.5:
                    # Store boxes by class
                    if class_name == 'personFF':
                        person_ff_boxes.append((bbox, conf))
                    elif class_name == 'personFB':
                        person_fb_boxes.append((bbox, conf))
                    elif class_name == 'counter':
                        counter_boxes.append((bbox, conf))
                    
                    # Draw bounding box
                    color = self.class_colors.get(class_name, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{class_name} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0] + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Check for overlap between personFF and counter
        overlap_detected = False
        max_overlap_iou = 0.0
        
        if person_ff_boxes and counter_boxes:
            for person_box, person_conf in person_ff_boxes:
                for counter_box, counter_conf in counter_boxes:
                    iou, intersection_area = self.calculate_iou(person_box, counter_box)
                    
                    if iou > 0 or intersection_area > 0:
                        overlap_detected = True
                        max_overlap_iou = max(max_overlap_iou, iou)
                        
                        # Draw line connecting overlapping boxes
                        p1_center = ((person_box[0] + person_box[2]) // 2, 
                                   (person_box[1] + person_box[3]) // 2)
                        c_center = ((counter_box[0] + counter_box[2]) // 2, 
                                  (counter_box[1] + counter_box[3]) // 2)
                        cv2.line(frame, p1_center, c_center, (0, 255, 0), 3)
                        
                        # Display overlap percentage
                        mid_point = ((p1_center[0] + c_center[0]) // 2,
                                   (p1_center[1] + c_center[1]) // 2)
                        cv2.putText(frame, f"Overlap: {iou*100:.1f}%", mid_point,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update timing based on overlap
        if overlap_detected:
            if not self.person_at_counter:
                # Person just arrived at counter
                self.person_at_counter = True
                self.counter_start_time = current_time
                self.counter_entries += 1
                self.current_session_time = 0.0
                print(f"\n[ENTRY {self.counter_entries}] PersonFF at counter (frame {self.frame_count})")
            else:
                # Person still at counter, update session time
                self.current_session_time = current_time - self.counter_start_time
        else:
            if self.person_at_counter:
                # Person left counter
                self.person_at_counter = False
                if self.counter_start_time:
                    session_duration = current_time - self.counter_start_time
                    self.total_time_at_counter += session_duration
                    print(f"[EXIT {self.counter_entries}] PersonFF left counter. Session: {session_duration:.2f}s")
                self.counter_start_time = None
                self.current_session_time = 0.0
        
        # Draw statistics
        self.draw_statistics(frame, person_ff_boxes, person_fb_boxes, 
                           counter_boxes, overlap_detected)
        
        return frame
    
    def draw_statistics(self, frame, person_ff_boxes, person_fb_boxes, 
                       counter_boxes, overlap_detected):
        """Draw statistics overlay on frame"""
        y_offset = 30
        line_height = 30
        
        # Background for statistics
        cv2.rectangle(frame, (10, 10), (450, 240), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 240), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "FINE-TUNED YOLO TRACKER", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height + 5
        
        # Detection counts
        stats = [
            f"FPS: {self.fps:.1f}",
            f"PersonFF: {len(person_ff_boxes)} | PersonFB: {len(person_fb_boxes)} | Counter: {len(counter_boxes)}",
            f"Counter Entries: {self.counter_entries}",
            f"Total Time: {self.total_time_at_counter:.2f}s",
            f"Current Session: {self.current_session_time:.2f}s",
            f"Status: {'AT COUNTER' if self.person_at_counter else 'Away from Counter'}"
        ]
        
        for i, stat in enumerate(stats):
            if i == len(stats) - 1:  # Status line
                color = (0, 255, 0) if self.person_at_counter else (128, 128, 128)
            else:
                color = (255, 255, 255)
            
            cv2.putText(frame, stat, (20, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    
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
        
        print("\nProcessing started...")
        print("Press 'Q' to quit")
        
        # Processing loop
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("\nEnd of video or camera disconnected")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Fine-tuned YOLO Tracker - Press Q to quit', processed_frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    print("\nQuitting...")
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            elapsed_time = time.time() - start_time
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self.print_final_statistics(elapsed_time)
    
    def print_final_statistics(self, elapsed_time):
        """Print comprehensive final statistics"""
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Elapsed time: {elapsed_time:.2f}s")
        print(f"Average FPS: {self.frame_count/elapsed_time:.2f}")
        print()
        print("COUNTER TRACKING:")
        print(f"  Total entries: {self.counter_entries}")
        print(f"  Total time at counter: {self.total_time_at_counter:.2f}s")
        if self.counter_entries > 0:
            print(f"  Average time per session: {self.total_time_at_counter/self.counter_entries:.2f}s")
        print("=" * 60)


def main():
    """Main entry point"""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "FINE-TUNED YOLO TRACKER" + " " * 25 + "║")
    print("║" + " " * 5 + "PersonFF | PersonFB | Counter Detection" + " " * 12 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Configuration
    model_path = "best.pt"  # Fine-tuned YOLO model
    
    # Choose video source
    # Option 1: Use webcam
    # video_path = None
    
    # Option 2: Use video file
    video_path = r"../data_manipulator/Data_sample_Time_processing_&_Emotion_Detection/sample_cam3.mp4"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Make sure 'best.pt' is in the fine_tuned_version directory")
        return
    
    # Check if video exists (if using video file)
    if video_path and not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    
    # Initialize and run tracker
    tracker = FineTunedYOLOTracker(model_path, video_path)
    tracker.run()


if __name__ == "__main__":
    main()
