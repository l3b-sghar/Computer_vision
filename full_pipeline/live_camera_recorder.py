"""
Live Camera Recording System with YOLO Person Detection
========================================================

This script captures live camera footage and automatically starts recording
when a person is detected by YOLO, saving individual session videos that
can be processed through the pipeline.

Features:
- Real-time person detection using YOLOv8
- Automatic session recording (start when person enters, stop when they leave)
- Configurable ROI (Region of Interest) for detection zone
- Session-based video files for pipeline processing
- FPS counter and visual feedback
- Graceful session management with timeout handling

Author: TALEL BOUSSETTA
Date: November 16, 2025
"""

import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO
import numpy as np


class LiveCameraRecorder:
    """
    Manages live camera recording with automatic person detection and session management.
    """
    
    def __init__(self, 
                 camera_index=0,
                 model_path='yolov8n.pt',
                 output_dir='recorded_sessions',
                 roi_height_ratio=0.3,
                 confidence_threshold=0.5,
                 no_person_timeout=3.0,
                 min_session_duration=2.0):
        """
        Initialize the live camera recorder.
        
        Args:
            camera_index: Camera device index (0 for default webcam)
            model_path: Path to YOLO model weights
            output_dir: Directory to save recorded sessions
            roi_height_ratio: Height ratio for ROI (0.3 = bottom 30% of frame)
            confidence_threshold: Minimum confidence for person detection
            no_person_timeout: Seconds to wait before ending session (person left)
            min_session_duration: Minimum session duration to save (seconds)
        """
        self.camera_index = camera_index
        self.model_path = model_path
        self.output_dir = output_dir
        self.roi_height_ratio = roi_height_ratio
        self.confidence_threshold = confidence_threshold
        self.no_person_timeout = no_person_timeout
        self.min_session_duration = min_session_duration
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize YOLO model
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        print("✓ YOLO model loaded successfully")
        
        # Session tracking
        self.recording = False
        self.video_writer = None
        self.session_start_time = None
        self.last_person_detected_time = None
        self.session_counter = 0
        self.current_session_path = None
        
        # Performance tracking
        self.fps_counter = []
        self.frame_count = 0
        
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            box1: (x1, y1, x2, y2) for first box
            box2: (x1, y1, x2, y2) for second box
            
        Returns:
            IoU value (0.0 to 1.0)
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def detect_person_in_roi(self, frame, roi_box):
        """
        Detect if a person is present in the ROI using YOLO.
        
        Args:
            frame: Current video frame
            roi_box: (x1, y1, x2, y2) coordinates of ROI
            
        Returns:
            Tuple of (person_detected: bool, detections: list)
        """
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        person_in_roi = False
        detections = []
        
        # Check each detection
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if it's a person (class 0 in COCO dataset)
                if int(box.cls[0]) == 0:
                    confidence = float(box.conf[0])
                    
                    if confidence >= self.confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        person_box = (x1, y1, x2, y2)
                        
                        # Calculate IoU with ROI
                        iou = self.calculate_iou(person_box, roi_box)
                        
                        detections.append({
                            'box': person_box,
                            'confidence': confidence,
                            'iou': iou
                        })
                        
                        # Consider person in ROI if IoU > 0 (any overlap)
                        if iou > 0:
                            person_in_roi = True
        
        return person_in_roi, detections
    
    def start_recording(self, frame_shape):
        """
        Start a new recording session.
        
        Args:
            frame_shape: (height, width, channels) of video frames
        """
        self.session_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{self.session_counter:03d}_{timestamp}.mp4"
        self.current_session_path = os.path.join(self.output_dir, filename)
        
        # Initialize video writer
        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # Target FPS for saved video
        
        self.video_writer = cv2.VideoWriter(
            self.current_session_path,
            fourcc,
            fps,
            (width, height)
        )
        
        self.recording = True
        self.session_start_time = time.time()
        self.last_person_detected_time = time.time()
        
        print(f"\n🔴 RECORDING STARTED - Session {self.session_counter}")
        print(f"   File: {filename}")
    
    def stop_recording(self):
        """
        Stop the current recording session.
        """
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        session_duration = time.time() - self.session_start_time if self.session_start_time else 0
        
        # Check if session meets minimum duration
        if session_duration < self.min_session_duration:
            print(f"⚠️  Session too short ({session_duration:.1f}s), deleting...")
            if os.path.exists(self.current_session_path):
                os.remove(self.current_session_path)
        else:
            print(f"⏹️  RECORDING STOPPED - Duration: {session_duration:.1f}s")
            print(f"   Saved: {self.current_session_path}")
        
        self.recording = False
        self.session_start_time = None
        self.last_person_detected_time = None
        self.current_session_path = None
    
    def draw_overlay(self, frame, roi_box, person_detected, detections):
        """
        Draw visual overlay on frame with ROI, detections, and status.
        
        Args:
            frame: Video frame to draw on
            roi_box: (x1, y1, x2, y2) coordinates of ROI
            person_detected: Whether person is in ROI
            detections: List of person detections
            
        Returns:
            Frame with overlay drawn
        """
        overlay = frame.copy()
        
        # Draw ROI
        roi_color = (0, 255, 0) if person_detected else (0, 165, 255)  # Green if person, orange otherwise
        cv2.rectangle(overlay, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), 
                     roi_color, 2)
        
        # Draw ROI label
        roi_label = "DETECTION ZONE" + (" - PERSON DETECTED" if person_detected else "")
        cv2.putText(overlay, roi_label, (roi_box[0] + 10, roi_box[1] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)
        
        # Draw person detections
        for detection in detections:
            box = detection['box']
            conf = detection['confidence']
            iou = detection['iou']
            
            # Different color based on whether person is in ROI
            color = (0, 255, 0) if iou > 0 else (255, 0, 0)
            
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Draw label with confidence and IoU
            label = f"Person {conf:.2f}"
            if iou > 0:
                label += f" (IoU: {iou:.2f})"
            
            cv2.putText(overlay, label, (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate FPS
        if len(self.fps_counter) > 0:
            avg_fps = 1.0 / (sum(self.fps_counter) / len(self.fps_counter))
        else:
            avg_fps = 0.0
        
        # Draw status panel
        panel_height = 120
        cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (400, panel_height), (255, 255, 255), 2)
        
        # Recording status
        status = "🔴 RECORDING" if self.recording else "⚪ STANDBY"
        status_color = (0, 0, 255) if self.recording else (200, 200, 200)
        cv2.putText(overlay, status, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Session info
        cv2.putText(overlay, f"Session: {self.session_counter}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS
        cv2.putText(overlay, f"FPS: {avg_fps:.1f}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Session timer
        if self.recording and self.session_start_time:
            duration = time.time() - self.session_start_time
            cv2.putText(overlay, f"Duration: {duration:.1f}s", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(overlay, "Press 'q' to quit", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay
    
    def run(self):
        """
        Main loop for live camera recording.
        """
        # Open camera
        print(f"\nOpening camera {self.camera_index}...")
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"❌ ERROR: Could not open camera {self.camera_index}")
            return
        
        # Get camera properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        camera_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✓ Camera opened successfully")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {camera_fps}")
        
        # Define ROI (bottom portion of frame)
        roi_y1 = int(frame_height * (1 - self.roi_height_ratio))
        roi_box = (0, roi_y1, frame_width, frame_height)
        
        print(f"\n📍 Detection Zone: Bottom {int(self.roi_height_ratio * 100)}% of frame")
        print(f"   ROI: y={roi_y1} to y={frame_height}")
        print(f"\n▶️  Starting live detection...")
        print(f"   Output directory: {self.output_dir}")
        print(f"   No-person timeout: {self.no_person_timeout}s")
        print(f"   Min session duration: {self.min_session_duration}s")
        print("\n" + "=" * 60)
        
        try:
            while True:
                frame_start_time = time.time()
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ ERROR: Failed to read frame")
                    break
                
                self.frame_count += 1
                
                # Detect person in ROI
                person_detected, detections = self.detect_person_in_roi(frame, roi_box)
                
                # Handle recording logic
                current_time = time.time()
                
                if person_detected:
                    self.last_person_detected_time = current_time
                    
                    # Start recording if not already recording
                    if not self.recording:
                        self.start_recording(frame.shape)
                
                # Check if we should stop recording
                if self.recording:
                    # Write frame to video
                    self.video_writer.write(frame)
                    
                    # Check timeout (person left)
                    if self.last_person_detected_time is not None:
                        time_since_last_detection = current_time - self.last_person_detected_time
                        
                        if time_since_last_detection > self.no_person_timeout:
                            self.stop_recording()
                
                # Draw overlay
                display_frame = self.draw_overlay(frame, roi_box, person_detected, detections)
                
                # Show frame
                cv2.imshow('Live Camera Recorder - YOLO Person Detection', display_frame)
                
                # Calculate FPS
                frame_time = time.time() - frame_start_time
                self.fps_counter.append(frame_time)
                if len(self.fps_counter) > 30:  # Keep last 30 frames for average
                    self.fps_counter.pop(0)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n\n🛑 Quit signal received")
                    break
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        
        finally:
            # Cleanup
            if self.recording:
                print("\nStopping current recording...")
                self.stop_recording()
            
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 60)
            print("📊 RECORDING SESSION SUMMARY")
            print("=" * 60)
            print(f"Total sessions recorded: {self.session_counter}")
            print(f"Output directory: {self.output_dir}")
            print(f"Total frames processed: {self.frame_count}")
            print("=" * 60)
            print("\n✓ Camera recorder closed successfully")


def main():
    """
    Main entry point for live camera recorder.
    """
    # Configuration
    config = {
        'camera_index': 0,                    # Default webcam
        'model_path': 'yolov8n.pt',          # YOLO model (will auto-download if not found)
        'output_dir': 'recorded_sessions',    # Output directory for session videos
        'roi_height_ratio': 0.3,             # Bottom 30% of frame
        'confidence_threshold': 0.5,          # YOLO confidence threshold
        'no_person_timeout': 3.0,            # 3 seconds after person leaves
        'min_session_duration': 2.0          # Minimum 2 seconds to save session
    }
    
    print("\n" + "=" * 60)
    print("🎥 LIVE CAMERA RECORDER WITH YOLO PERSON DETECTION")
    print("=" * 60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Create and run recorder
    recorder = LiveCameraRecorder(**config)
    recorder.run()


if __name__ == "__main__":
    main()
