"""
YOLO Model Test Demo - Live Camera

This script tests the YOLO model (yolo11s.pt) for object detection on live camera feed.
This is a baseline test before fine-tuning for custom emotion/interest detection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cv2
import numpy as np
from collections import deque, Counter

# Try to import ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("=" * 60)
    print("ERROR: ultralytics not found")
    print("=" * 60)
    print("\nPlease install ultralytics:")
    print("  pip install ultralytics")
    print("\nThis package provides YOLOv8/v11 models")
    print("=" * 60)
    import sys
    sys.exit(1)


class YOLOTestDemo:
    """Test YOLO model on live camera feed."""
    
    def __init__(self, model_path, camera_id=0, confidence_threshold=0.5):
        """
        Initialize the demo.
        
        Args:
            model_path: Path to YOLO model (.pt file)
            camera_id: Camera device ID (default: 0)
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        
        # Statistics tracking
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.detection_counts = Counter()
        self.start_time = None
        
        # Model and camera
        self.model = None
        self.cap = None
        
    def load_model(self):
        """Load the YOLO model."""
        print(f"Loading YOLO model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model file not found at {self.model_path}")
            return False
        
        try:
            self.model = YOLO(self.model_path)
            print(f"✓ YOLO model loaded successfully")
            print(f"  Model type: {self.model.model.__class__.__name__}")
            
            # Get model info
            if hasattr(self.model, 'names'):
                print(f"  Classes: {len(self.model.names)} total")
                print(f"  Sample classes: {list(self.model.names.values())[:5]}...")
            
            return True
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize camera capture."""
        print(f"Opening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"ERROR: Could not open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✓ Camera {self.camera_id} opened")
        return True
    
    def detect_objects(self, frame):
        """
        Run YOLO detection on frame.
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            YOLO results object
        """
        # Run inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        return results[0]  # Get first result (single image)
    
    def draw_detections(self, frame, results):
        """
        Draw detection results on frame.
        
        Args:
            frame: Input frame
            results: YOLO results object
            
        Returns:
            Annotated frame
        """
        # Get detections
        boxes = results.boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                # Update statistics
                self.detection_counts[class_name] += 1
                
                # Choose color based on class
                color = self._get_color_for_class(cls)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f"{class_name}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw label background
                cv2.rectangle(frame, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _get_color_for_class(self, cls_id):
        """Get consistent color for class ID."""
        # Generate color based on class ID
        np.random.seed(cls_id)
        color = tuple(map(int, np.random.randint(50, 255, 3)))
        return color
    
    def draw_info_panel(self, frame, results):
        """Draw information panel on frame."""
        h, w = frame.shape[:2]
        panel_height = 120
        
        # Semi-transparent panel at bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # FPS
        if len(self.fps_history) > 0:
            avg_fps = np.mean(self.fps_history)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, h - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Frame count
        cv2.putText(frame, f"Frames: {self.frame_count}", (10, h - 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Detection count in current frame
        num_detections = len(results.boxes) if results.boxes is not None else 0
        cv2.putText(frame, f"Detections: {num_detections}", (10, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'Q' to quit, 'R' to reset, 'S' for stats", 
                   (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Elapsed time
        if self.start_time:
            elapsed = time.time() - self.start_time
            cv2.putText(frame, f"Time: {elapsed:.1f}s", (200, h - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Model confidence threshold
        cv2.putText(frame, f"Confidence: {self.confidence_threshold:.2f}", (200, h - 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def print_statistics(self):
        """Print session statistics."""
        print("\n" + "=" * 50)
        print("SESSION STATISTICS")
        print("=" * 50)
        print(f"Total frames processed: {self.frame_count}")
        
        if len(self.fps_history) > 0:
            print(f"Average FPS: {np.mean(self.fps_history):.2f}")
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"Total time: {elapsed:.1f}s")
        
        # Detection statistics
        total_detections = sum(self.detection_counts.values())
        if total_detections > 0:
            print(f"\nTotal detections: {total_detections}")
            print("\nTop detected classes:")
            for class_name, count in self.detection_counts.most_common(10):
                percentage = (count / total_detections) * 100
                bar_length = int(percentage / 2)
                bar = "█" * bar_length
                print(f"  {class_name:15}: {count:5} ({percentage:5.1f}%) {bar}")
        else:
            print("\nNo detections recorded")
        
        print("=" * 50 + "\n")
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.frame_count = 0
        self.detection_counts.clear()
        self.fps_history.clear()
        self.start_time = time.time()
        print("Statistics reset")
    
    def run(self):
        """Run the live demo."""
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + " " * 18 + "YOLO TEST DEMO" + " " * 26 + "║")
        print("╚" + "═" * 58 + "╝\n")
        
        print("Testing YOLO model for object detection")
        print(f"Model: {os.path.basename(self.model_path)}\n")
        
        print("Initializing YOLO Test Demo...")
        print("-" * 50)
        
        # Load model
        if not self.load_model():
            return
        
        # Initialize camera
        if not self.initialize_camera():
            return
        
        print("-" * 50)
        print("\nStarting live camera feed...")
        print("Press 'Q' to quit, 'R' to reset, 'S' to show statistics")
        print(f"Confidence threshold: {self.confidence_threshold}\n")
        
        self.start_time = time.time()
        
        try:
            while True:
                frame_start = time.time()
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Run YOLO detection
                results = self.detect_objects(frame)
                
                # Draw detections
                frame = self.draw_detections(frame, results)
                frame = self.draw_info_panel(frame, results)
                
                # Update frame count and FPS
                self.frame_count += 1
                frame_time = time.time() - frame_start
                if frame_time > 0:
                    self.fps_history.append(1.0 / frame_time)
                
                # Display frame
                cv2.imshow('YOLO Test Demo - Object Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r') or key == ord('R'):
                    self.reset_statistics()
                elif key == ord('s') or key == ord('S'):
                    self.print_statistics()
                elif key == ord('+') or key == ord('='):
                    self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self.print_statistics()
            
            print("Demo ended")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Model Test Demo')
    parser.add_argument('--model', type=str, 
                       default='../models/yolo11s.pt',
                       help='Path to YOLO model file (.pt)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (0.0-1.0, default: 0.5)')
    
    args = parser.parse_args()
    
    # Resolve model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model)
    
    # Create and run demo
    demo = YOLOTestDemo(
        model_path=model_path, 
        camera_id=args.camera,
        confidence_threshold=args.conf
    )
    demo.run()


if __name__ == '__main__':
    main()
