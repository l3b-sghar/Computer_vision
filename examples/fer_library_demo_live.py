"""
FER Library Demo - Live Camera

This script demonstrates real-time facial emotion recognition using the 'fer' library.
The 'fer' library is a lightweight alternative to DeepFace with MTCNN face detection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import cv2
from collections import deque

# Try to import FER library (correct import path)
try:
    from fer.fer import FER
    FER_AVAILABLE = True
except ImportError as e:
    FER_AVAILABLE = False
    print("=" * 60)
    print("ERROR: 'fer' library not found or cannot be imported")
    print(f"Details: {e}")
    print("=" * 60)
    print("\nPlease install the fer library:")
    print("  pip install fer")
    print("\nOr try:")
    print("  pip uninstall fer")
    print("  pip install fer")
    print("\nAlternatively, you can use the DeepFace version:")
    print("  pip install deepface")
    print("  # Then modify fer.py to use DeepFace")
    print("=" * 60)
    import sys
    sys.exit(1)


class FERLibraryDemo:
    """Live facial emotion recognition demo using the fer library."""
    
    def __init__(self, camera_index=0):
        """
        Initialize FER library demo.
        
        Args:
            camera_index: Camera index (0 for default webcam)
        """
        self.camera_index = camera_index
        self.detector = None
        self.cap = None
        
        # Statistics
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        
        # Emotion tracking
        self.emotion_history = []
        
        # Colors for visualization
        self.emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'surprise': (0, 255, 255),  # Yellow
            'neutral': (255, 255, 255), # White
            'sad': (255, 0, 0),         # Blue
            'angry': (0, 0, 255),       # Red
            'disgust': (128, 0, 128),   # Purple
            'fear': (0, 165, 255)       # Orange
        }
    
    def initialize(self) -> bool:
        """
        Initialize FER detector and camera.
        
        Returns:
            True if initialization successful
        """
        print("Initializing FER Library Demo...")
        print("-" * 50)
        
        # Initialize FER detector
        # mtcnn=True uses MTCNN for better face detection (slower but more accurate)
        # mtcnn=False uses OpenCV Haar Cascade (faster but less accurate)
        try:
            print("Loading FER detector (this may take a moment)...")
            self.detector = FER(mtcnn=False)  # Set to True for better accuracy
            print("✓ FER detector initialized")
        except Exception as e:
            print(f"✗ Failed to initialize FER detector: {e}")
            return False
        
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"✗ Cannot open camera {self.camera_index}")
            return False
        print(f"✓ Camera {self.camera_index} opened")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("-" * 50)
        print()
        return True
    
    def draw_emotion_bars(self, frame, emotions, x, y, width=200):
        """
        Draw emotion probability bars.
        
        Args:
            frame: Frame to draw on
            emotions: Dictionary of emotion probabilities
            x, y: Top-left position for bars
            width: Width of bars
        """
        bar_height = 15
        spacing = 5
        
        # Sort emotions by probability
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion, prob) in enumerate(sorted_emotions):
            y_pos = y + i * (bar_height + spacing)
            
            # Background bar
            cv2.rectangle(frame, (x, y_pos), (x + width, y_pos + bar_height), 
                         (50, 50, 50), -1)
            
            # Probability bar
            bar_width = int(width * prob)
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y_pos), (x + bar_width, y_pos + bar_height), 
                         color, -1)
            
            # Text label
            text = f"{emotion}: {prob:.2f}"
            cv2.putText(frame, text, (x + width + 10, y_pos + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_face_annotations(self, frame, face_data):
        """
        Draw annotations for detected face.
        
        Args:
            frame: Frame to draw on
            face_data: Dictionary containing face box and emotions
        """
        box = face_data['box']
        emotions = face_data['emotions']
        
        x, y, w, h = box
        
        # Get dominant emotion
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        emotion = dominant_emotion[0]
        confidence = dominant_emotion[1]
        
        # Get color
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw emotion label
        label = f"{emotion.upper()} ({confidence:.2f})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Background for label
        cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                     (x + label_size[0] + 10, y), color, -1)
        
        # Label text
        cv2.putText(frame, label, (x + 5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw emotion probability bars if space available
        bar_x = x + w + 10
        bar_y = y
        if bar_x + 250 < frame.shape[1]:
            self.draw_emotion_bars(frame, emotions, bar_x, bar_y)
    
    def draw_stats(self, frame, fps, face_count):
        """
        Draw statistics overlay.
        
        Args:
            frame: Frame to draw on
            fps: Current FPS
            face_count: Number of faces detected
        """
        height, width = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Statistics text
        cv2.putText(frame, "FER LIBRARY DEMO", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Faces: {face_count}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'Q' to quit | 'R' to reset | 'S' for stats", 
                   (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def calculate_fps(self):
        """Calculate current FPS."""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if self.last_time else 0.0
        self.last_time = current_time
        
        self.fps_history.append(fps)
        return np.mean(self.fps_history) if self.fps_history else 0.0
    
    def show_statistics(self):
        """Show detailed statistics."""
        print("\n" + "=" * 50)
        print("SESSION STATISTICS")
        print("=" * 50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Average FPS: {np.mean(self.fps_history):.2f}" if self.fps_history else "N/A")
        
        if len(self.emotion_history) > 0:
            print(f"\nEmotion Statistics:")
            
            # Count each emotion
            emotion_counts = {}
            for emotions in self.emotion_history:
                dominant = max(emotions.items(), key=lambda x: x[1])[0]
                emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1
            
            # Calculate percentages
            total = len(self.emotion_history)
            print(f"\nEmotion Distribution:")
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100
                bar = "█" * int(percentage / 2)
                print(f"  {emotion:10s}: {percentage:5.1f}% {bar}")
        
        print("=" * 50)
    
    def run(self):
        """Run the live FER demo."""
        if not self.initialize():
            print("Failed to initialize demo")
            return
        
        print("Starting live camera feed...")
        print("Press 'Q' to quit, 'R' to reset, 'S' to show statistics")
        print()
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                self.frame_count += 1
                
                # Detect emotions using FER library
                # Returns list of detected faces with emotions
                result = self.detector.detect_emotions(frame)
                
                # Process each detected face
                for face in result:
                    self.emotion_history.append(face['emotions'])
                    self.draw_face_annotations(frame, face)
                
                # Calculate FPS
                fps = self.calculate_fps()
                
                # Draw stats overlay
                self.draw_stats(frame, fps, len(result))
                
                # Display frame
                cv2.imshow('FER Library Demo', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r') or key == ord('R'):
                    print("\nResetting statistics...")
                    self.emotion_history = []
                    self.frame_count = 0
                    self.fps_history.clear()
                elif key == ord('s') or key == ord('S'):
                    self.show_statistics()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            self.show_statistics()
            self.cap.release()
            cv2.destroyAllWindows()
            print("\nDemo ended")


def main():
    """Main entry point."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "FER LIBRARY DEMO" + " " * 27 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("This demo uses the 'fer' library for emotion recognition")
    print("Install: pip install fer")
    print()
    
    # Create and run demo
    demo = FERLibraryDemo(camera_index=0)
    demo.run()


if __name__ == "__main__":
    main()
