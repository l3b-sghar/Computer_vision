"""
Live Camera FER Demo

This script demonstrates real-time facial emotion recognition using your webcam.
It uses OpenCV's Haar Cascade for face detection and the FER module for emotion recognition.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import cv2
from config import Config
from analytics import FacialEmotionRecognizer


class LiveFERDemo:
    """Live facial emotion recognition demo using webcam."""
    
    def __init__(self, camera_index=0):
        """
        Initialize live FER demo.
        
        Args:
            camera_index: Camera index (0 for default webcam)
        """
        self.camera_index = camera_index
        self.fer = None
        self.face_cascade = None
        self.cap = None
        
        # Statistics
        self.frame_count = 0
        self.fps_history = []
        self.last_time = time.time()
        
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
        Initialize FER module and face detector.
        
        Returns:
            True if initialization successful
        """
        print("Initializing Live FER Demo...")
        print("-" * 50)
        
        # Initialize FER
        self.fer = FacialEmotionRecognizer(Config)
        if not self.fer.initialize():
            print("Warning: FER initialized in fallback mode")
        else:
            print("✓ FER module initialized")
        
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("✗ Failed to load face detector")
            return False
        print("✓ Face detector loaded")
        
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
    
    def detect_faces(self, frame):
        """
        Detect faces in frame using Haar Cascade.
        
        Args:
            frame: Input frame
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        return faces
    
    def draw_emotion_bar(self, frame, emotion_probs, x, y, width=200):
        """
        Draw emotion probability bars.
        
        Args:
            frame: Frame to draw on
            emotion_probs: Dictionary of emotion probabilities
            x, y: Top-left position for bars
            width: Width of bars
        """
        bar_height = 15
        spacing = 5
        
        # Sort emotions by probability
        sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
        
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
    
    def draw_face_annotations(self, frame, face_bbox, emotion_result):
        """
        Draw annotations for detected face.
        
        Args:
            frame: Frame to draw on
            face_bbox: Face bounding box (x, y, w, h)
            emotion_result: Emotion recognition result
        """
        x, y, w, h = face_bbox
        
        # Get emotion and color
        emotion = emotion_result['emotion']
        confidence = emotion_result['confidence']
        satisfaction = emotion_result['satisfaction_score_face']
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
        
        # Draw satisfaction score
        satisfaction_text = f"Satisfaction: {satisfaction:.2f}"
        cv2.putText(frame, satisfaction_text, (x, y + h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw emotion probability bars
        bar_x = x + w + 10
        bar_y = y
        if bar_x + 250 < frame.shape[1]:  # If space available on right
            self.draw_emotion_bar(frame, emotion_result['probabilities'], bar_x, bar_y)
    
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
        cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Statistics text
        cv2.putText(frame, "LIVE FER DEMO", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Faces: {face_count}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Frames: {self.frame_count}", (20, 95),
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
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        return np.mean(self.fps_history) if self.fps_history else 0.0
    
    def show_statistics(self):
        """Show detailed statistics."""
        print("\n" + "=" * 50)
        print("SESSION STATISTICS")
        print("=" * 50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Average FPS: {np.mean(self.fps_history):.2f}" if self.fps_history else "N/A")
        
        if len(self.fer.emotion_history) > 0:
            avg_result = self.fer.compute_average_emotion(window_size=len(self.fer.emotion_history))
            print(f"\nEmotion Statistics:")
            print(f"  Dominant emotion: {avg_result['dominant_emotion']}")
            print(f"  Average confidence: {avg_result['average_confidence']:.3f}")
            print(f"  Average satisfaction: {avg_result['average_satisfaction']:.3f}")
            print(f"\nEmotion Distribution:")
            for emotion, prob in sorted(avg_result['emotion_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 30)
                print(f"  {emotion:10s}: {prob:.3f} {bar}")
        print("=" * 50)
    
    def run(self):
        """Run the live FER demo."""
        if not self.initialize():
            print("Failed to initialize demo")
            return
        
        print("Starting live camera feed...")
        print("Press 'Q' to quit, 'R' to reset FER, 'S' to show statistics")
        print()
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                self.frame_count += 1
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process each face
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_crop = frame[y:y+h, x:x+w]
                    
                    # Recognize emotion
                    emotion_result = self.fer.recognize_emotion(
                        face_crop, 
                        use_temporal_smoothing=True
                    )
                    
                    # Draw annotations
                    self.draw_face_annotations(frame, (x, y, w, h), emotion_result)
                
                # Calculate FPS
                fps = self.calculate_fps()
                
                # Draw stats overlay
                self.draw_stats(frame, fps, len(faces))
                
                # Display frame
                cv2.imshow('Live FER Demo', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r') or key == ord('R'):
                    print("\nResetting FER state...")
                    self.fer.reset()
                    self.frame_count = 0
                    self.fps_history = []
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
    print("║" + " " * 15 + "LIVE CAMERA FER DEMO" + " " * 23 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Create and run demo
    demo = LiveFERDemo(camera_index=0)
    demo.run()


if __name__ == "__main__":
    main()
