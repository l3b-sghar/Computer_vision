"""
Full Pipeline: YOLO ROI Tracker + FER Emotion Detection + Body Language Analysis

This script combines:
1. YOLO person detection with ROI time tracking
2. FER emotion detection during ROI presence
3. MediaPipe body language analysis (posture, movement)
4. Returns: (processing_time, average_emotional_state, body_language_score)
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
import os
import sys
import math
import collections

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import FER library
try:
    from fer.fer import FER
    FER_AVAILABLE = True
except ImportError as e:
    FER_AVAILABLE = False
    print("=" * 60)
    print("ERROR: 'fer' library not found")
    print(f"Details: {e}")
    print("Please install: pip install fer")
    print("=" * 60)
    sys.exit(1)

# Try to import TensorFlow Lite for body language
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("=" * 60)
    print("WARNING: TensorFlow not found - body language analysis disabled")
    print("Install with: pip install tensorflow")
    print("=" * 60)

# MediaPipe is not used in main.py (this uses main_simple.py instead)
MEDIAPIPE_AVAILABLE = False


class IntegratedPipeline:
    """Integrated YOLO ROI tracking with FER emotion detection and body language analysis."""
    
    def __init__(self, yolo_model_path, video_path=None, yolo_skip_frames=5, fer_skip_frames=5, pose_skip_frames=5):
        """
        Initialize the integrated pipeline.
        
        Args:
            yolo_model_path: Path to YOLO model
            video_path: Path to video file (None for webcam)
            yolo_skip_frames: Process YOLO every N frames (5 for performance)
            fer_skip_frames: Process FER every N frames (5 for performance)
            pose_skip_frames: Process pose every N frames (5 for performance)
        """
        self.yolo_model = YOLO(yolo_model_path)
        self.fer_detector = FER(mtcnn=False)  # FER emotion detector
        self.video_path = video_path
        
        # MediaPipe Pose for body language (not available in this version)
        self.pose_detector = None
        
        # Performance optimization
        self.yolo_skip_frames = yolo_skip_frames
        self.fer_skip_frames = fer_skip_frames
        self.pose_skip_frames = pose_skip_frames
        self.last_yolo_result = None
        self.last_fer_result = None
        self.last_pose_result = None
        
        # Region of Interest (bottom 1/5th of frame)
        self.roi_height_ratio = 0.2
        
        # ROI tracking variables
        self.person_in_roi = False
        self.roi_start_time = None
        self.total_time_in_roi = 0.0
        self.current_session_time = 0.0
        self.roi_entries = 0
        
        # Emotion tracking (only when person in ROI)
        self.emotions_in_roi = []  # List of emotion dictionaries
        self.emotion_history = []  # All emotions for visualization
        
        # Body language tracking
        self.landmark_history = collections.deque(maxlen=6)
        self.body_scores_in_roi = []
        self.prev_body_score = None
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()
        
        # Emotion colors for visualization
        self.emotion_colors = {
            'happy': (0, 255, 0),
            'surprise': (0, 255, 255),
            'neutral': (255, 255, 255),
            'sad': (255, 0, 0),
            'angry': (0, 0, 255),
            'disgust': (128, 0, 128),
            'fear': (0, 165, 255)
        }
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes."""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0, 0
        
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou, intersection_area
    
    def calculate_average_emotion(self, emotion_list):
        """
        Calculate average emotional state from a list of emotion dictionaries.
        
        Args:
            emotion_list: List of emotion dictionaries
        
        Returns:
            Dictionary with average probabilities for each emotion
        """
        if not emotion_list:
            return None
        
        # Sum all emotion probabilities
        emotion_sum = {}
        for emotions in emotion_list:
            for emotion, prob in emotions.items():
                emotion_sum[emotion] = emotion_sum.get(emotion, 0) + prob
        
        # Calculate averages
        num_samples = len(emotion_list)
        emotion_avg = {emotion: prob / num_samples 
                      for emotion, prob in emotion_sum.items()}
        
        return emotion_avg
    
    def get_dominant_emotion(self, emotion_dict):
        """Get the dominant emotion from emotion dictionary."""
        if not emotion_dict:
            return "unknown", 0.0
        return max(emotion_dict.items(), key=lambda x: x[1])
    
    def _angle_between_deg(self, v1, v2):
        """Calculate angle between two vectors in degrees."""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return math.degrees(math.acos(c))
    
    def _clamp01(self, x):
        """Clamp value between 0 and 1."""
        return max(0.0, min(1.0, x))
    
    def _map_to_01(self, v, a, b):
        """Map value from range [a, b] to [0, 1]."""
        if b == a:
            return 0.0
        return self._clamp01((v - a) / (b - a))
    
    def compute_body_language_score(self, history, frame_w, frame_h, prev_score=None, alpha_smooth=0.6):
        """
        Compute body language satisfaction score from pose landmarks history.
        
        Returns:
            (label, score, diagnostics)
        """
        if len(history) < 3:
            return None, 0, {}
        
        pts_now = history[-1]
        pts_prev = history[-2]
        pts_old = history[-3]
        
        if pts_now.shape[0] < 25:
            return None, 0, {}
        
        # Calculate activity (movement)
        disp1 = np.linalg.norm(pts_now - pts_prev, axis=1)
        disp2 = np.linalg.norm(pts_prev - pts_old, axis=1)
        activity = float((disp1.mean() + disp2.mean()) / 2.0)
        
        # Landmark indices
        L_SH, R_SH = 11, 12
        L_HIP, R_HIP = 23, 24
        NOSE = 0
        L_WRIST, R_WRIST = 15, 16
        
        # Key points
        shoulders = (pts_now[L_SH] + pts_now[R_SH]) / 2.0
        hips = (pts_now[L_HIP] + pts_now[R_HIP]) / 2.0
        nose = pts_now[NOSE]
        
        # Torso alignment
        torso_vec = hips - shoulders
        torso_len = np.linalg.norm(torso_vec) + 1e-6
        torso_dir = torso_vec / torso_len
        
        vertical = np.array([0.0, 1.0])
        torso_angle = self._angle_between_deg(torso_dir, vertical)
        torso_angle = min(torso_angle, 180 - torso_angle)
        
        # Head alignment
        head_vec = nose - shoulders
        head_len = np.linalg.norm(head_vec) + 1e-6
        head_dir = head_vec / head_len
        head_torso_angle = self._angle_between_deg(head_dir, torso_dir)
        head_torso_angle = min(head_torso_angle, 180 - head_torso_angle)
        
        # Shoulder symmetry
        sh_ys = abs(pts_now[L_SH][1] - pts_now[R_SH][1])
        shoulder_sym = sh_ys * frame_h / (torso_len * frame_h + 1e-6)
        
        # Arm openness
        shoulder_width = np.linalg.norm(pts_now[L_SH] - pts_now[R_SH]) + 1e-6
        wrist_dist = np.linalg.norm(pts_now[L_WRIST] - pts_now[R_WRIST])
        arm_openness = self._clamp01((wrist_dist / shoulder_width) / 2.5)
        
        # Hands near face (stress indicator)
        wrist_to_nose = min(np.linalg.norm(pts_now[L_WRIST] - nose),
                           np.linalg.norm(pts_now[R_WRIST] - nose))
        hands_face = self._map_to_01(wrist_to_nose, 0.01, 0.20)
        
        # Calculate sub-scores
        activity_score = self._map_to_01(activity, 0.0008, 0.018)
        
        if torso_angle <= 10:
            upright_score = 1.0
        elif torso_angle >= 40:
            upright_score = 0.0
        else:
            upright_score = 1.0 - ((torso_angle - 10) / (40 - 10))
        
        head_align_score = self._clamp01(1.0 - (head_torso_angle / 40.0))
        shoulder_sym_score = 1.0 - self._clamp01(shoulder_sym * 3.0)
        hands_open_score = hands_face
        
        # Combined score
        combined = (
            0.35 * upright_score +
            0.30 * activity_score +
            0.15 * hands_open_score +
            0.12 * head_align_score +
            0.08 * shoulder_sym_score
        )
        combined = self._clamp01(combined)
        
        # Smooth with previous score
        if prev_score is not None:
            combined = alpha_smooth * combined + (1.0 - alpha_smooth) * (prev_score / 100.0)
        
        score = int(combined * 100)
        
        if score >= 70:
            label = "satisfied"
        elif score >= 45:
            label = "neutral"
        else:
            label = "dissatisfied"
        
        diag = {
            "activity": activity,
            "torso_angle": torso_angle,
            "head_torso_angle": head_torso_angle,
            "activity_score": activity_score,
            "upright_score": upright_score,
            "head_align_score": head_align_score,
            "shoulder_sym_score": shoulder_sym_score,
            "combined_raw": combined
        }
        
        return label, score, diag
    
    def process_frame(self, frame):
        """Process a single frame with YOLO + FER (optimized with frame skipping)."""
        self.frame_count += 1
        height, width = frame.shape[:2]
        
        # Calculate FPS
        current_time = time.time()
        if current_time - self.fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.fps_time)
            self.fps_time = current_time
            self.frame_count = 0
        
        # Define ROI (bottom 1/5th of frame)
        roi_y1 = int(height * (1 - self.roi_height_ratio))
        roi_box = (0, roi_y1, width, height)
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (0, roi_y1), (width, height), (0, 255, 255), 3)
        cv2.putText(frame, "Region of Interest", (10, roi_y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Run YOLO person detection (with frame skipping)
        person_detected_in_roi = False
        person_roi_box = None  # Store the person box that's in ROI
        
        if self.frame_count % self.yolo_skip_frames == 0:
            yolo_results = self.yolo_model(frame, verbose=False)
            self.last_yolo_result = yolo_results
        else:
            yolo_results = self.last_yolo_result
        
        # Process YOLO detections and find person in ROI
        if yolo_results:
            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Filter for person class (class 0)
                    if cls == 0 and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        person_box = (x1, y1, x2, y2)
                        
                        # Calculate overlap with ROI
                        iou, intersection_area = self.calculate_iou(person_box, roi_box)
                        
                        # Check if person is in ROI
                        if iou > 0 or intersection_area > 0:
                            color = (0, 255, 0)  # Green
                            if not person_detected_in_roi:  # Only track the first person in ROI
                                person_detected_in_roi = True
                                person_roi_box = person_box
                            label = f"Person {conf:.2f} [IN ROI]"
                        else:
                            color = (255, 0, 0)  # Blue
                            label = f"Person {conf:.2f}"
                        
                        # Draw person bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Run MediaPipe Pose detection ONLY for person in ROI
        current_body_label = None
        current_body_score = 0
        if self.pose_detector and person_detected_in_roi and person_roi_box and self.frame_count % self.pose_skip_frames == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose_detector.process(frame_rgb)
            self.last_pose_result = pose_results
        elif person_detected_in_roi and self.last_pose_result:
            pose_results = self.last_pose_result
        else:
            pose_results = None
            self.last_pose_result = None
        
        # Process pose results and compute body language score
        if pose_results and pose_results.pose_landmarks and person_detected_in_roi:
            lm = pose_results.pose_landmarks.landmark
            pts = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
            self.landmark_history.append(pts)
            
            # Draw pose landmarks on person
            if person_roi_box:
                x1, y1, x2, y2 = person_roi_box
                for p in lm:
                    px = int(p.x * width)
                    py = int(p.y * height)
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        cv2.circle(frame, (px, py), 3, (255, 100, 0), -1)
            
            # Compute body language score
            current_body_label, current_body_score, body_diag = self.compute_body_language_score(
                self.landmark_history, width, height, 
                prev_score=self.prev_body_score, alpha_smooth=0.6
            )
            
            if current_body_label and person_detected_in_roi:
                self.body_scores_in_roi.append(current_body_score)
                self.prev_body_score = current_body_score
        
        # Run FER emotion detection ONLY for the person in ROI
        current_emotion = None
        if person_detected_in_roi and person_roi_box and self.frame_count % self.fer_skip_frames == 0:
            # Crop the frame to only the person's bounding box in ROI
            x1, y1, x2, y2 = person_roi_box
            # Expand slightly to ensure face is included
            padding = 20
            y1_crop = max(0, y1 - padding)
            y2_crop = min(height, y2 + padding)
            x1_crop = max(0, x1 - padding)
            x2_crop = min(width, x2 + padding)
            
            person_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            
            if person_crop.size > 0:
                fer_results = self.fer_detector.detect_emotions(person_crop)
                # Adjust face coordinates back to full frame
                if fer_results:
                    for face in fer_results:
                        face['box'] = (
                            face['box'][0] + x1_crop,
                            face['box'][1] + y1_crop,
                            face['box'][2],
                            face['box'][3]
                        )
                self.last_fer_result = fer_results
            else:
                fer_results = None
        elif person_detected_in_roi and self.last_fer_result:
            fer_results = self.last_fer_result
        else:
            fer_results = None
            self.last_fer_result = None  # Clear when no person in ROI
        
        # Process FER results (only one face - the person in ROI)
        if fer_results and person_detected_in_roi and len(fer_results) > 0:
            # Get the first (and only relevant) face detected
            face = fer_results[0]
            emotions = face['emotions']
            box = face['box']
            
            # Store emotions for averaging
            self.emotions_in_roi.append(emotions)
            self.emotion_history.append(emotions)
            current_emotion = emotions
            
            # Draw face box and emotion
            fx, fy, fw, fh = box
            dominant_emotion, confidence = self.get_dominant_emotion(emotions)
            emotion_color = self.emotion_colors.get(dominant_emotion, (255, 255, 255))
            
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), emotion_color, 2)
            
            emotion_label = f"{dominant_emotion.upper()} ({confidence:.2f})"
            cv2.putText(frame, emotion_label, (fx, fy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 2)
        
        # Update ROI timing
        current_time = time.time()
        
        if person_detected_in_roi:
            if not self.person_in_roi:
                # Person just entered ROI
                self.person_in_roi = True
                self.roi_start_time = current_time
                self.roi_entries += 1
                self.current_session_time = 0.0
                print(f"\n[ENTRY {self.roi_entries}] Person entered ROI at frame {self.frame_count}")
            else:
                # Person still in ROI
                self.current_session_time = current_time - self.roi_start_time
        else:
            if self.person_in_roi:
                # Person just left ROI
                self.person_in_roi = False
                if self.roi_start_time:
                    session_duration = current_time - self.roi_start_time
                    self.total_time_in_roi += session_duration
                    print(f"[EXIT {self.roi_entries}] Person left ROI. Session duration: {session_duration:.2f}s")
                self.roi_start_time = None
                self.current_session_time = 0.0
        
        # Draw comprehensive statistics
        self.draw_statistics(frame, current_emotion, current_body_label, current_body_score)
        
        return frame
    
    def draw_statistics(self, frame, current_emotion, current_body_label, current_body_score):
        """Draw statistics overlay on frame."""
        y_offset = 30
        line_height = 25
        
        # Background for statistics
        cv2.rectangle(frame, (10, 10), (450, 360), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 360), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "INTEGRATED PIPELINE", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height + 5
        
        # ROI Statistics
        stats = [
            f"FPS: {self.fps:.1f}",
            f"ROI Entries: {self.roi_entries}",
            f"Total Time in ROI: {self.total_time_in_roi:.2f}s",
            f"Current Session: {self.current_session_time:.2f}s",
            f"Status: {'IN ROI' if self.person_in_roi else 'Outside ROI'}"
        ]
        
        for i, stat in enumerate(stats):
            color = (0, 255, 0) if self.person_in_roi and i == 4 else (255, 255, 255)
            cv2.putText(frame, stat, (20, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        y_offset += len(stats) * line_height + 10
        
        # Body Language Statistics
        cv2.putText(frame, "BODY LANGUAGE (IN ROI)", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += line_height
        
        if MEDIAPIPE_AVAILABLE and current_body_label:
            cv2.putText(frame, f"Current: {current_body_label.upper()} ({current_body_score})", 
                       (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            if self.body_scores_in_roi:
                avg_body = int(np.mean(self.body_scores_in_roi))
                cv2.putText(frame, f"Average: {avg_body}", 
                           (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += line_height
        elif not MEDIAPIPE_AVAILABLE:
            cv2.putText(frame, "MediaPipe not available", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            y_offset += line_height
        
        # Emotion Statistics
        cv2.putText(frame, "FACE EMOTION (IN ROI)", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += line_height
        
        if current_emotion:
            # Show current emotion
            dominant, conf = self.get_dominant_emotion(current_emotion)
            cv2.putText(frame, f"Current: {dominant.upper()} ({conf:.2f})", 
                       (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
        
        # Show average emotion (if we have data)
        if self.emotions_in_roi:
            avg_emotion = self.calculate_average_emotion(self.emotions_in_roi)
            dominant_avg, conf_avg = self.get_dominant_emotion(avg_emotion)
            cv2.putText(frame, f"Average: {dominant_avg.upper()} ({conf_avg:.2f})", 
                       (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "No emotions detected yet", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    def run(self, display=True):
        """
        Run the integrated pipeline.
        
        Args:
            display: Whether to display video window
        
        Returns:
            Tuple of (processing_time, average_emotional_state)
        """
        # Open video source
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            print(f"Processing video: {self.video_path}")
        else:
            cap = cv2.VideoCapture(0)
            print("Opening webcam...")
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return None, None
        
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
                if display:
                    cv2.imshow('Integrated Pipeline - Press Q to quit', processed_frame)
                    
                    # Handle key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("\nQuitting...")
                        break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Calculate results
            processing_time = self.total_time_in_roi
            average_emotion = self.calculate_average_emotion(self.emotions_in_roi)
            average_body_score = int(np.mean(self.body_scores_in_roi)) if self.body_scores_in_roi else 0
            
            # Cleanup
            elapsed_time = time.time() - start_time
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            # Print final statistics
            self.print_final_statistics(elapsed_time, processing_time, average_emotion, average_body_score)
            
            return processing_time, average_emotion, average_body_score
    
    def print_final_statistics(self, elapsed_time, processing_time, average_emotion, average_body_score):
        """Print comprehensive final statistics."""
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Elapsed time: {elapsed_time:.2f}s")
        print(f"Average processing FPS: {self.frame_count/elapsed_time:.2f}")
        print()
        print("ROI TRACKING:")
        print(f"  ROI entries: {self.roi_entries}")
        print(f"  Total time person in ROI: {processing_time:.2f}s")
        if self.roi_entries > 0:
            print(f"  Average time per session: {processing_time/self.roi_entries:.2f}s")
        print()
        print("BODY LANGUAGE ANALYSIS:")
        print(f"  Total body samples (in ROI): {len(self.body_scores_in_roi)}")
        if self.body_scores_in_roi:
            print(f"  Average body language score: {average_body_score}")
            if average_body_score >= 70:
                body_label = "satisfied"
            elif average_body_score >= 45:
                body_label = "neutral"
            else:
                body_label = "dissatisfied"
            print(f"  Body language state: {body_label}")
        else:
            print("  No body language data")
        print()
        print("FACE EMOTION ANALYSIS:")
        print(f"  Total emotion samples (in ROI): {len(self.emotions_in_roi)}")
        
        if average_emotion:
            dominant_emotion, confidence = self.get_dominant_emotion(average_emotion)
            print(f"  Dominant emotion: {dominant_emotion.upper()} ({confidence:.2%})")
            print()
            print("  Emotion distribution:")
            sorted_emotions = sorted(average_emotion.items(), 
                                   key=lambda x: x[1], reverse=True)
            for emotion, prob in sorted_emotions:
                bar = "█" * int(prob * 50)
                print(f"    {emotion:10s}: {prob:6.2%} {bar}")
        else:
            print("  No emotions detected")
        
        print("=" * 60)
        print()
        print("RETURN VALUES:")
        print(f"  Processing Time: {processing_time:.2f}s")
        if average_emotion:
            dominant, conf = self.get_dominant_emotion(average_emotion)
            print(f"  Average Emotion State: {dominant} ({conf:.2%})")
        else:
            print(f"  Average Emotion State: None")
        print(f"  Average Body Language Score: {average_body_score}")
        print("=" * 60)


def main():
    """Main entry point."""
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + " " * 15 + "INTEGRATED PIPELINE" + " " * 36 + "║")
    print("║" + " " * 5 + "YOLO ROI + FER Emotion + Body Language Analysis" + " " * 17 + "║")
    print("╚" + "═" * 70 + "╝")
    print()
    
    if not MEDIAPIPE_AVAILABLE:
        print("⚠️  MediaPipe not available - body language analysis will be disabled")
        print("   Install with: pip install mediapipe (requires Python 3.8-3.12)")
        print()
    
    # Configuration
    yolo_model_path = r"../models/yolo11s.pt"
    
    # Choose video source
    # Option 1: Use webcam
    # video_path = None
    
    # Option 2: Use video file
    video_path = r"../data_manipulator/Data_sample_Time_processing_&_Emotion_Detection/sample_cam1.mp4"
    
    # Check if model exists
    if not os.path.exists(yolo_model_path):
        print(f"Error: YOLO model not found at {yolo_model_path}")
        return
    
    # Check if video exists (if using video file)
    if video_path and not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    
    # Initialize and run pipeline (all models run every 5 frames for performance)
    pipeline = IntegratedPipeline(yolo_model_path, video_path, 
                                 yolo_skip_frames=5, 
                                 fer_skip_frames=5, 
                                 pose_skip_frames=5)
    processing_time, average_emotion, average_body_score = pipeline.run(display=True)
    
    # Return values
    print("\nRETURN VALUES:")
    print(f"  processing_time = {processing_time:.2f} seconds")
    if average_emotion:
        dominant, confidence = max(average_emotion.items(), key=lambda x: x[1])
        print(f"  average_emotion = '{dominant}' (confidence: {confidence:.2%})")
        print(f"  Full emotion distribution: {average_emotion}")
    else:
        print(f"  average_emotion = None (no emotions detected)")
    print(f"  average_body_language_score = {average_body_score}")



if __name__ == "__main__":
    main()
