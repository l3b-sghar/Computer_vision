"""
FER Custom Model Demo - Live Camera

This script uses the custom trained model (ferplus_model_pd_best.h5) from the models folder
for real-time facial emotion recognition from webcam feed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import cv2
from collections import deque

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    import json
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("=" * 60)
    print("ERROR: TensorFlow not found")
    print("=" * 60)
    print("\nPlease install TensorFlow:")
    print("  pip install tensorflow")
    print("=" * 60)
    import sys
    sys.exit(1)


class CustomFERDemo:
    """Live camera emotion detection using custom trained model."""
    
    def __init__(self, model_path, camera_id=0):
        """
        Initialize the demo.
        
        Args:
            model_path: Path to the trained model (.h5 file)
            camera_id: Camera device ID (default: 0)
        """
        self.model_path = model_path
        self.camera_id = camera_id
        
        # Emotion labels for FER+ model (8 emotions)
        self.emotion_labels = ['neutral', 'happiness', 'surprise', 'sadness', 
                               'anger', 'disgust', 'fear', 'contempt']
        
        # Color mapping for emotions (BGR format)
        self.emotion_colors = {
            'neutral': (200, 200, 200),      # Gray
            'happiness': (0, 255, 0),        # Green
            'surprise': (255, 255, 0),       # Cyan
            'sadness': (255, 0, 0),          # Blue
            'anger': (0, 0, 255),            # Red
            'disgust': (0, 128, 128),        # Dark Yellow
            'fear': (128, 0, 128),           # Purple
            'contempt': (128, 128, 0)        # Teal
        }
        
        # Statistics tracking
        self.frame_count = 0
        self.emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
        self.fps_history = deque(maxlen=30)
        self.start_time = None
        
        # Temporal smoothing
        self.smoothed_probs = None
        self.smoothing_alpha = 0.6
        
        # Model and face detector
        self.model = None
        self.face_cascade = None
        self.cap = None
        
    def load_model(self):
        """Load the custom trained model."""
        print(f"Loading custom model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model file not found at {self.model_path}")
            return False
        
        # Try multiple loading strategies
        strategies = [
            ("Standard load (compile=False)", lambda: keras.models.load_model(self.model_path, compile=False)),
            ("Load with safe_mode=False", lambda: keras.models.load_model(self.model_path, compile=False, safe_mode=False)),
        ]
        
        for strategy_name, load_fn in strategies:
            try:
                print(f"Trying: {strategy_name}...")
                self.model = load_fn()
                
                # Recompile the model
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                print(f"✓ Model loaded successfully using {strategy_name}")
                print(f"  Input shape: {self.model.input_shape}")
                print(f"  Output shape: {self.model.output_shape}")
                return True
            except Exception as e:
                print(f"  Failed: {str(e)[:100]}...")
                continue
        
        print("\nAll loading strategies failed.")
        print("This model appears to have compatibility issues with the current TensorFlow version.")
        print("\nTry:")
        print("  1. Use DeepFace or fer library demos instead")
        print("  2. Re-save the model with current TensorFlow version")
        return False
    
    def initialize_detector(self):
        """Initialize face detector."""
        print("Loading face detector...")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("ERROR: Could not load face cascade classifier")
            return False
        
        print("✓ Face detector initialized")
        return True
    
    def initialize_camera(self):
        """Initialize camera capture."""
        print(f"Opening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"ERROR: Could not open camera {self.camera_id}")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✓ Camera {self.camera_id} opened")
        return True
    
    def preprocess_face(self, face_img):
        """
        Preprocess face image for model input.
        
        Args:
            face_img: Face crop (BGR image)
            
        Returns:
            Preprocessed image ready for model
        """
        # Convert to grayscale (FER+ models typically use grayscale)
        if len(face_img.shape) == 3:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_img
        
        # Resize to model input size (typically 64x64 for FER+)
        input_shape = self.model.input_shape[1:3]
        face_resized = cv2.resize(face_gray, input_shape)
        
        # Normalize to [0, 1]
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Add channel dimension if needed
        if len(self.model.input_shape) == 4 and self.model.input_shape[-1] == 1:
            face_normalized = np.expand_dims(face_normalized, axis=-1)
        elif len(self.model.input_shape) == 4 and self.model.input_shape[-1] == 3:
            # Convert back to 3 channels if model expects RGB
            face_normalized = cv2.cvtColor(face_normalized, cv2.COLOR_GRAY2RGB)
        
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def detect_emotions(self, frame):
        """
        Detect faces and recognize emotions.
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            List of tuples (bbox, emotion, confidence, probabilities)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )
        
        results = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess and predict
            face_input = self.preprocess_face(face_roi)
            predictions = self.model.predict(face_input, verbose=0)[0]
            
            # Apply temporal smoothing
            if self.smoothed_probs is None:
                self.smoothed_probs = predictions
            else:
                self.smoothed_probs = (self.smoothing_alpha * predictions + 
                                      (1 - self.smoothing_alpha) * self.smoothed_probs)
            
            # Get emotion with highest probability
            emotion_idx = np.argmax(self.smoothed_probs)
            emotion = self.emotion_labels[emotion_idx]
            confidence = self.smoothed_probs[emotion_idx]
            
            results.append(((x, y, w, h), emotion, confidence, self.smoothed_probs))
        
        return results
    
    def draw_results(self, frame, results):
        """
        Draw detection results on frame.
        
        Args:
            frame: Input frame
            results: Detection results
            
        Returns:
            Annotated frame
        """
        for (x, y, w, h), emotion, confidence, probabilities in results:
            # Get color for this emotion
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw emotion label
            label = f"{emotion}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y-label_size[1]-10), (x+label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            # Draw emotion probabilities bar chart
            bar_height = 15
            bar_width = 200
            bar_x = x + w + 10
            bar_y_start = y
            
            # Only draw bars if there's space
            if bar_x + bar_width < frame.shape[1]:
                for i, (emotion_name, prob) in enumerate(zip(self.emotion_labels, probabilities)):
                    bar_y = bar_y_start + i * (bar_height + 5)
                    
                    # Background bar
                    cv2.rectangle(frame, (bar_x, bar_y), 
                                (bar_x + bar_width, bar_y + bar_height), 
                                (50, 50, 50), -1)
                    
                    # Probability bar
                    bar_length = int(bar_width * prob)
                    bar_color = self.emotion_colors.get(emotion_name, (255, 255, 255))
                    cv2.rectangle(frame, (bar_x, bar_y), 
                                (bar_x + bar_length, bar_y + bar_height), 
                                bar_color, -1)
                    
                    # Label
                    cv2.putText(frame, f"{emotion_name[:3]}: {prob:.2f}", 
                              (bar_x + 5, bar_y + 12), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_info_panel(self, frame):
        """Draw information panel on frame."""
        h, w = frame.shape[:2]
        panel_height = 100
        
        # Semi-transparent panel at bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # FPS
        if len(self.fps_history) > 0:
            avg_fps = np.mean(self.fps_history)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, h - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Frame count
        cv2.putText(frame, f"Frames: {self.frame_count}", (10, h - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'Q' to quit, 'R' to reset, 'S' for stats", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Elapsed time
        if self.start_time:
            elapsed = time.time() - self.start_time
            cv2.putText(frame, f"Time: {elapsed:.1f}s", (200, h - 70),
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
        
        # Emotion distribution
        total_detections = sum(self.emotion_counts.values())
        if total_detections > 0:
            print("\nEmotion Statistics:")
            print("\nEmotion Distribution:")
            for emotion in sorted(self.emotion_counts.keys(), 
                                 key=lambda x: self.emotion_counts[x], reverse=True):
                count = self.emotion_counts[emotion]
                percentage = (count / total_detections) * 100
                bar_length = int(percentage / 2)
                bar = "█" * bar_length
                print(f"  {emotion:12}: {percentage:5.1f}% {bar}")
        
        print("=" * 50 + "\n")
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.frame_count = 0
        self.emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
        self.fps_history.clear()
        self.smoothed_probs = None
        self.start_time = time.time()
        print("Statistics reset")
    
    def run(self):
        """Run the live demo."""
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + " " * 15 + "CUSTOM FER MODEL DEMO" + " " * 22 + "║")
        print("╚" + "═" * 58 + "╝\n")
        
        print("Using custom trained model from models folder")
        print(f"Model: {os.path.basename(self.model_path)}\n")
        
        print("Initializing Custom FER Demo...")
        print("-" * 50)
        
        # Load model
        if not self.load_model():
            return
        
        # Initialize face detector
        if not self.initialize_detector():
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
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect emotions
                results = self.detect_emotions(frame)
                
                # Update statistics
                for _, emotion, _, _ in results:
                    self.emotion_counts[emotion] += 1
                
                # Draw results
                frame = self.draw_results(frame, results)
                frame = self.draw_info_panel(frame)
                
                # Update frame count and FPS
                self.frame_count += 1
                frame_time = time.time() - frame_start
                if frame_time > 0:
                    self.fps_history.append(1.0 / frame_time)
                
                # Display frame
                cv2.imshow('Custom FER Model Demo - Live Camera', frame)
                
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
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self.print_statistics()
            
            print("Demo ended")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Custom FER Model Live Demo')
    parser.add_argument('--model', type=str, 
                       default='../models/ferplus_model_pd_best.h5',
                       help='Path to trained model file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Resolve model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model)
    
    # Create and run demo
    demo = CustomFERDemo(model_path=model_path, camera_id=args.camera)
    demo.run()


if __name__ == '__main__':
    main()
