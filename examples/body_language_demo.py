"""
Body Language Analysis Demo - Live Camera

This script analyzes body language and posture for engagement detection.
Uses pose estimation to detect:
- Posture (standing/sitting/leaning)
- Body orientation (facing camera or turned away)
- Engagement indicators (head position, shoulder alignment)

Version 0 - Basic implementation with MediaPipe Pose
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cv2
import numpy as np
from collections import deque
import math

# Try to import MediaPipe (optional)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Note: MediaPipe not available, using simplified body language analysis")
    print("For full pose estimation, install mediapipe (requires Python 3.8-3.12)")
    print("Continuing with face-based engagement detection...\n")


class BodyLanguageAnalyzer:
    """Analyze body language and engagement from pose estimation."""
    
    def __init__(self, camera_id=0):
        """
        Initialize the analyzer.
        
        Args:
            camera_id: Camera device ID (default: 0)
        """
        self.camera_id = camera_id
        
        # MediaPipe pose detector (if available)
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.pose = None
        else:
            self.mp_pose = None
            self.pose = None
        
        # Face detector for simplified mode
        self.face_cascade = None
        
        # Engagement metrics
        self.engagement_history = deque(maxlen=30)  # Last 30 frames
        self.posture_history = deque(maxlen=30)
        
        # Statistics
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.start_time = None
        self.posture_counts = {'standing': 0, 'sitting': 0, 'leaning_forward': 0, 
                               'leaning_back': 0, 'unknown': 0}
        self.engagement_scores = []
        
        # Camera
        self.cap = None
        
    def initialize_pose_detector(self):
        """Initialize pose detector (MediaPipe or face-based fallback)."""
        if MEDIAPIPE_AVAILABLE and self.mp_pose:
            print("Loading MediaPipe Pose detector...")
            try:
                self.pose = self.mp_pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_complexity=1  # 0=lite, 1=full, 2=heavy
                )
                print("✓ Pose detector initialized (MediaPipe)")
                return True
            except Exception as e:
                print(f"WARNING: MediaPipe initialization failed: {e}")
                print("Falling back to simplified mode...")
        
        # Fallback: Use face detection for engagement estimation
        print("Loading face detector for simplified body language analysis...")
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("✓ Face detector initialized (Simplified mode)")
            print("  Note: Using face size/position as proxy for engagement")
            return True
        except Exception as e:
            print(f"ERROR initializing face detector: {e}")
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
    
    def analyze_pose(self, landmarks):
        """
        Analyze body language from pose landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            dict with posture, engagement_score, and metrics
        """
        if not landmarks:
            return {
                'posture': 'unknown',
                'engagement_score': 0.0,
                'body_angle': 0.0,
                'head_tilt': 0.0,
                'shoulder_alignment': 0.0,
                'facing_camera': False
            }
        
        # Extract key landmarks
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate metrics
        posture = self._classify_posture(nose, left_shoulder, right_shoulder, left_hip, right_hip)
        body_angle = self._calculate_body_angle(left_shoulder, right_shoulder, left_hip, right_hip)
        head_tilt = self._calculate_head_tilt(nose, left_shoulder, right_shoulder)
        shoulder_alignment = self._calculate_shoulder_alignment(left_shoulder, right_shoulder)
        facing_camera = self._is_facing_camera(left_shoulder, right_shoulder)
        
        # Calculate engagement score (0-1)
        engagement_score = self._calculate_engagement_score(
            posture, body_angle, head_tilt, shoulder_alignment, facing_camera
        )
        
        return {
            'posture': posture,
            'engagement_score': engagement_score,
            'body_angle': body_angle,
            'head_tilt': head_tilt,
            'shoulder_alignment': shoulder_alignment,
            'facing_camera': facing_camera
        }
    
    def _classify_posture(self, nose, l_shoulder, r_shoulder, l_hip, r_hip):
        """Classify posture based on body position."""
        # Calculate average shoulder and hip positions
        shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
        hip_y = (l_hip.y + r_hip.y) / 2
        nose_y = nose.y
        
        # Calculate body lean
        body_lean = nose_y - shoulder_y
        torso_length = hip_y - shoulder_y
        
        if torso_length == 0:
            return 'unknown'
        
        lean_ratio = body_lean / torso_length
        
        # Classify posture
        if lean_ratio < -0.15:
            return 'leaning_forward'  # Engaged/interested
        elif lean_ratio > 0.15:
            return 'leaning_back'  # Relaxed/disengaged
        elif hip_y < 0.7:  # Hips high in frame = standing
            return 'standing'
        else:
            return 'sitting'
    
    def _calculate_body_angle(self, l_shoulder, r_shoulder, l_hip, r_hip):
        """Calculate body lean angle (forward/backward tilt)."""
        shoulder_center_y = (l_shoulder.y + r_shoulder.y) / 2
        hip_center_y = (l_hip.y + r_hip.y) / 2
        
        # Vertical alignment (1.0 = perfectly upright)
        vertical_diff = abs(hip_center_y - shoulder_center_y)
        return min(vertical_diff * 10, 1.0)  # Scale to 0-1
    
    def _calculate_head_tilt(self, nose, l_shoulder, r_shoulder):
        """Calculate head tilt relative to shoulders."""
        shoulder_center_x = (l_shoulder.x + r_shoulder.x) / 2
        head_offset = abs(nose.x - shoulder_center_x)
        return head_offset
    
    def _calculate_shoulder_alignment(self, l_shoulder, r_shoulder):
        """Calculate shoulder alignment (0 = level, higher = tilted)."""
        shoulder_diff = abs(l_shoulder.y - r_shoulder.y)
        return shoulder_diff
    
    def _is_facing_camera(self, l_shoulder, r_shoulder):
        """Determine if person is facing camera based on shoulder visibility."""
        # If both shoulders visible and roughly same depth, facing camera
        shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        # Typical shoulder width when facing camera is 0.2-0.4 in normalized coords
        return 0.15 < shoulder_width < 0.5
    
    def _calculate_engagement_score(self, posture, body_angle, head_tilt, 
                                    shoulder_alignment, facing_camera):
        """
        Calculate overall engagement score (0-1).
        
        Higher score = more engaged
        """
        score = 0.5  # Base score
        
        # Posture contribution
        if posture == 'leaning_forward':
            score += 0.3  # Very engaged
        elif posture == 'standing':
            score += 0.1  # Neutral
        elif posture == 'sitting':
            score += 0.0  # Neutral
        elif posture == 'leaning_back':
            score -= 0.2  # Less engaged
        
        # Facing camera (important)
        if facing_camera:
            score += 0.2
        else:
            score -= 0.3
        
        # Head alignment (centered = engaged)
        if head_tilt < 0.1:
            score += 0.1
        elif head_tilt > 0.2:
            score -= 0.1
        
        # Shoulder alignment (level = attentive)
        if shoulder_alignment < 0.05:
            score += 0.05
        
        # Clamp to 0-1
        return max(0.0, min(1.0, score))
    
    def _analyze_face_engagement(self, face_bbox, frame_shape):
        """
        Simplified engagement analysis using face detection.
        
        Args:
            face_bbox: (x, y, w, h) face bounding box
            frame_shape: (height, width, channels) of frame
            
        Returns:
            dict with engagement metrics
        """
        x, y, w, h = face_bbox
        frame_h, frame_w = frame_shape[:2]
        
        # Calculate metrics
        face_area = w * h
        frame_area = frame_h * frame_w
        face_ratio = face_area / frame_area
        
        # Face position (centered = more engaged)
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        frame_center_x = frame_w / 2
        frame_center_y = frame_h / 2
        
        x_offset = abs(face_center_x - frame_center_x) / frame_w
        y_offset = abs(face_center_y - frame_center_y) / frame_h
        
        # Calculate engagement
        # Large face + centered = high engagement
        size_score = min(face_ratio * 20, 1.0)  # Normalize
        center_score = 1.0 - (x_offset + y_offset) / 2
        
        engagement_score = (size_score * 0.6 + center_score * 0.4)
        
        # Classify posture based on face size
        if face_ratio > 0.15:
            posture = 'leaning_forward'
        elif face_ratio < 0.05:
            posture = 'leaning_back'
        elif y < frame_h * 0.3:
            posture = 'standing'
        else:
            posture = 'sitting'
        
        return {
            'posture': posture,
            'engagement_score': engagement_score,
            'body_angle': size_score,
            'head_tilt': x_offset,
            'shoulder_alignment': 0.0,
            'facing_camera': True
        }
    
    def draw_pose(self, frame, results):
        """Draw pose landmarks on frame."""
        if MEDIAPIPE_AVAILABLE and hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame
    
    def draw_analysis(self, frame, analysis):
        """Draw body language analysis on frame."""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for info panel
        overlay = frame.copy()
        panel_width = 300
        panel_height = 200
        cv2.rectangle(overlay, (w - panel_width - 10, 10), 
                     (w - 10, panel_height + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw analysis info
        x_offset = w - panel_width
        y_offset = 30
        line_height = 25
        
        # Title
        cv2.putText(frame, "BODY LANGUAGE", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += line_height + 5
        
        # Posture
        posture_label = analysis['posture'].replace('_', ' ').title()
        cv2.putText(frame, f"Posture: {posture_label}", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Engagement score with color coding
        engagement = analysis['engagement_score']
        if engagement > 0.7:
            color = (0, 255, 0)  # Green - highly engaged
            label = "HIGH"
        elif engagement > 0.4:
            color = (0, 255, 255)  # Yellow - moderate
            label = "MEDIUM"
        else:
            color = (0, 0, 255)  # Red - low engagement
            label = "LOW"
        
        cv2.putText(frame, f"Engagement: {engagement:.2f} ({label})", 
                   (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y_offset += line_height
        
        # Engagement bar
        bar_width = 250
        bar_height = 20
        bar_x = x_offset + 10
        bar_y = y_offset
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Fill based on engagement
        fill_width = int(bar_width * engagement)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        y_offset += line_height + 10
        
        # Facing camera indicator
        facing_text = "✓ Facing Camera" if analysis['facing_camera'] else "✗ Not Facing"
        facing_color = (0, 255, 0) if analysis['facing_camera'] else (0, 0, 255)
        cv2.putText(frame, facing_text, (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, facing_color, 1)
        y_offset += line_height
        
        # Additional metrics (small text)
        cv2.putText(frame, f"Body Angle: {analysis['body_angle']:.2f}", 
                   (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += line_height - 5
        
        cv2.putText(frame, f"Head Tilt: {analysis['head_tilt']:.2f}", 
                   (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def draw_info_panel(self, frame):
        """Draw information panel at bottom."""
        h, w = frame.shape[:2]
        panel_height = 80
        
        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # FPS
        if len(self.fps_history) > 0:
            avg_fps = np.mean(self.fps_history)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, h - 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Frame count
        cv2.putText(frame, f"Frames: {self.frame_count}", (10, h - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Average engagement
        if len(self.engagement_history) > 0:
            avg_engagement = np.mean(self.engagement_history)
            cv2.putText(frame, f"Avg Engagement: {avg_engagement:.2f}", 
                       (150, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'Q' to quit, 'R' to reset, 'S' for stats", 
                   (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
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
        
        # Engagement statistics
        if len(self.engagement_scores) > 0:
            print(f"\nAverage engagement score: {np.mean(self.engagement_scores):.2f}")
            print(f"Max engagement: {max(self.engagement_scores):.2f}")
            print(f"Min engagement: {min(self.engagement_scores):.2f}")
        
        # Posture distribution
        total_postures = sum(self.posture_counts.values())
        if total_postures > 0:
            print("\nPosture Distribution:")
            for posture in sorted(self.posture_counts.keys(), 
                                 key=lambda x: self.posture_counts[x], reverse=True):
                count = self.posture_counts[posture]
                percentage = (count / total_postures) * 100
                bar_length = int(percentage / 2)
                bar = "█" * bar_length
                posture_label = posture.replace('_', ' ').title()
                print(f"  {posture_label:20}: {percentage:5.1f}% {bar}")
        
        print("=" * 50 + "\n")
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.frame_count = 0
        self.engagement_history.clear()
        self.posture_history.clear()
        self.engagement_scores.clear()
        self.posture_counts = {k: 0 for k in self.posture_counts}
        self.fps_history.clear()
        self.start_time = time.time()
        print("Statistics reset")
    
    def run(self):
        """Run the live demo."""
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + " " * 15 + "BODY LANGUAGE DEMO" + " " * 25 + "║")
        print("╚" + "═" * 58 + "╝\n")
        
        print("Analyzing body language and engagement from pose estimation")
        print("Version 0 - Basic implementation\n")
        
        print("Initializing Body Language Analyzer...")
        print("-" * 50)
        
        # Initialize pose detector
        if not self.initialize_pose_detector():
            return
        
        # Initialize camera
        if not self.initialize_camera():
            return
        
        print("-" * 50)
        print("\nStarting live camera feed...")
        print("Press 'Q' to quit, 'R' to reset, 'S' to show statistics\n")
        
        self.start_time = time.time()
        
        try:
            while True:
                frame_start = time.time()
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process with MediaPipe if available
                if self.pose and MEDIAPIPE_AVAILABLE:
                    # Convert to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(rgb_frame)
                    
                    # Analyze body language
                    if results.pose_landmarks:
                        analysis = self.analyze_pose(results.pose_landmarks.landmark)
                        
                        # Update statistics
                        self.posture_counts[analysis['posture']] += 1
                        self.engagement_history.append(analysis['engagement_score'])
                        self.engagement_scores.append(analysis['engagement_score'])
                        
                        # Draw pose and analysis
                        frame = self.draw_pose(frame, results)
                        frame = self.draw_analysis(frame, analysis)
                    else:
                        cv2.putText(frame, "No person detected", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Fallback: Use face detection
                elif self.face_cascade:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                    
                    if len(faces) > 0:
                        # Use face size and position as proxy for engagement
                        analysis = self._analyze_face_engagement(faces[0], frame.shape)
                        
                        # Update statistics
                        self.posture_counts[analysis['posture']] += 1
                        self.engagement_history.append(analysis['engagement_score'])
                        self.engagement_scores.append(analysis['engagement_score'])
                        
                        # Draw face box and analysis
                        x, y, w, h = faces[0]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        frame = self.draw_analysis(frame, analysis)
                    else:
                        cv2.putText(frame, "No face detected", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Draw info panel
                frame = self.draw_info_panel(frame)
                
                # Update frame count and FPS
                self.frame_count += 1
                frame_time = time.time() - frame_start
                if frame_time > 0:
                    self.fps_history.append(1.0 / frame_time)
                
                # Display frame
                cv2.imshow('Body Language Analysis Demo', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r') or key == ord('R'):
                    self.reset_statistics()
                elif key == ord('s') or key == ord('S'):
                    self.print_statistics()
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            if self.pose:
                self.pose.close()
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self.print_statistics()
            
            print("Demo ended")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Body Language Analysis Demo')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Create and run analyzer
    analyzer = BodyLanguageAnalyzer(camera_id=args.camera)
    analyzer.run()


if __name__ == '__main__':
    main()
