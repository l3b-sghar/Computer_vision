"""
Body Language TFLite Model Demo - Live Camera

This script uses a TensorFlow Lite model (body_language.tflite) for body language
classification from live camera feed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cv2
import numpy as np
from collections import deque, Counter

# Try to import TensorFlow Lite
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("=" * 60)
    print("ERROR: TensorFlow not found")
    print("=" * 60)
    print("\nPlease install TensorFlow:")
    print("  pip install tensorflow")
    print("=" * 60)
    import sys
    sys.exit(1)


class BodyLanguageTFLiteDemo:
    """Body language classification using TFLite model."""
    
    def __init__(self, model_path, camera_id=0):
        """
        Initialize the demo.
        
        Args:
            model_path: Path to TFLite model file
            camera_id: Camera device ID (default: 0)
        """
        self.model_path = model_path
        self.camera_id = camera_id
        
        # TFLite interpreter
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        # Model info
        self.input_shape = None
        # Classes based on your training: 10 distinct body language and facial expressions
        self.output_classes = ['Happy', 'Sad', 'Angry', 'Surprised', 'Confused', 
                               'Tension', 'Excited', 'Pain', 'Depressed']
        
        # Pose estimation for input
        self.face_cascade = None
        
        # Statistics
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.class_counts = Counter()
        self.start_time = None
        self.confidence_history = deque(maxlen=30)
        
        # Camera
        self.cap = None
        
    def load_model(self):
        """Load the TFLite model."""
        print(f"Loading TFLite model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model file not found at {self.model_path}")
            return False
        
        try:
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get input shape
            self.input_shape = self.input_details[0]['shape']
            
            print(f"✓ TFLite model loaded successfully")
            print(f"  Input shape: {self.input_shape}")
            print(f"  Input dtype: {self.input_details[0]['dtype']}")
            print(f"  Output shape: {self.output_details[0]['shape']}")
            print(f"  Expected classes: {len(self.output_classes)}")
            print(f"\n  NOTE: This model expects MediaPipe Pose landmarks (2004 features)")
            print(f"  Currently using pixel sampling as workaround")
            print(f"  For accurate results, extract real pose landmarks with MediaPipe\n")
            
            return True
        except Exception as e:
            print(f"ERROR loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def initialize_face_detector(self):
        """Initialize face detector."""
        print("Loading face detector...")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("ERROR: Could not load face cascade")
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
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✓ Camera {self.camera_id} opened")
        return True
    
    def extract_pose_landmarks(self, frame):
        """
        Extract pose landmarks from frame.
        This model expects pose landmark features, not raw pixels.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Flattened landmark array (2004 features = 501 landmarks × 4 coords each)
            Or None if no pose detected
        """
        # For now, return dummy features since we don't have pose estimation
        # In production, you'd use MediaPipe Pose or similar
        
        # Expected shape is [1, 2004]
        # This suggests 501 landmarks with (x, y, z, visibility) = 4 values each
        # Or possibly 668 landmarks with 3 coords each
        
        num_features = self.input_shape[1]
        
        # Generate dummy features for demonstration
        # In real use, extract from MediaPipe Pose landmarks
        dummy_features = np.zeros(num_features, dtype=np.float32)
        
        # Add some variation based on frame to show it's "working"
        # In production, these would be real pose landmarks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Sample some pixel values as features (not ideal, but for demo)
        step = max(1, (h * w) // num_features)
        samples = gray.flatten()[::step][:num_features]
        dummy_features[:len(samples)] = samples.astype(np.float32) / 255.0
        
        return dummy_features
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for model input.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Preprocessed data ready for model
        """
        # This model expects pose landmarks, not image pixels
        features = self.extract_pose_landmarks(frame)
        
        if features is None:
            # Return zeros if no pose detected
            features = np.zeros(self.input_shape[1], dtype=np.float32)
        
        # Add batch dimension [1, num_features]
        input_data = np.expand_dims(features, axis=0)
        
        return input_data
    
    def predict(self, frame):
        """
        Run inference on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            (predicted_class, confidence, all_probabilities)
        """
        # Preprocess
        input_data = self.preprocess_frame(frame)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        probabilities = output_data[0]
        
        # Get predicted class
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        
        # Handle case where model has different number of outputs
        if len(probabilities) != len(self.output_classes):
            print(f"Warning: Model outputs {len(probabilities)} classes, expected {len(self.output_classes)}")
            # Pad or truncate output_classes
            if len(probabilities) > len(self.output_classes):
                self.output_classes = [f"class_{i}" for i in range(len(probabilities))]
        
        predicted_class = self.output_classes[predicted_idx] if predicted_idx < len(self.output_classes) else f"class_{predicted_idx}"
        
        return predicted_class, confidence, probabilities
    
    def draw_results(self, frame, predicted_class, confidence, probabilities):
        """
        Draw prediction results on frame.
        
        Args:
            frame: Input frame
            predicted_class: Predicted class name
            confidence: Confidence score
            probabilities: All class probabilities
            
        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        
        # Create info panel
        overlay = frame.copy()
        panel_width = 300
        panel_height = min(200, 40 + len(probabilities) * 30)
        cv2.rectangle(overlay, (w - panel_width - 10, 10),
                     (w - 10, panel_height + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw prediction
        x_offset = w - panel_width + 10
        y_offset = 35
        
        # Title
        cv2.putText(frame, "BODY LANGUAGE", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30
        
        # Main prediction with color coding
        if confidence > 0.7:
            color = (0, 255, 0)  # High confidence - green
        elif confidence > 0.4:
            color = (0, 255, 255)  # Medium - yellow
        else:
            color = (0, 165, 255)  # Low - orange
        
        pred_text = f"{predicted_class.upper()}"
        cv2.putText(frame, pred_text, (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 25
        
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 30
        
        # Draw probability bars
        bar_width = 250
        bar_height = 15
        
        for i, (class_name, prob) in enumerate(zip(self.output_classes[:len(probabilities)], probabilities)):
            # Class name
            cv2.putText(frame, class_name[:10], (x_offset, y_offset + bar_height - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Bar background
            bar_x = x_offset + 80
            cv2.rectangle(frame, (bar_x, y_offset),
                         (bar_x + bar_width - 80, y_offset + bar_height),
                         (50, 50, 50), -1)
            
            # Bar fill
            fill_width = int((bar_width - 80) * prob)
            bar_color = (0, 255, 0) if i == np.argmax(probabilities) else (100, 100, 100)
            cv2.rectangle(frame, (bar_x, y_offset),
                         (bar_x + fill_width, y_offset + bar_height),
                         bar_color, -1)
            
            # Probability value
            cv2.putText(frame, f"{prob:.2f}", (bar_x + bar_width - 70, y_offset + bar_height - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            y_offset += bar_height + 8
        
        return frame
    
    def draw_info_panel(self, frame):
        """Draw information panel at bottom."""
        h, w = frame.shape[:2]
        panel_height = 100
        
        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # FPS
        if len(self.fps_history) > 0:
            avg_fps = np.mean(self.fps_history)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, h - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Frame count
        cv2.putText(frame, f"Frames: {self.frame_count}", (10, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Average confidence
        if len(self.confidence_history) > 0:
            avg_conf = np.mean(self.confidence_history)
            cv2.putText(frame, f"Avg Conf: {avg_conf:.2f}", (150, h - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Model info
        cv2.putText(frame, f"Model: TFLite", (150, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'Q' to quit, 'R' to reset, 'S' for stats",
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Time
        if self.start_time:
            elapsed = time.time() - self.start_time
            cv2.putText(frame, f"Time: {elapsed:.1f}s", (10, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
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
        
        if len(self.confidence_history) > 0:
            print(f"Average confidence: {np.mean(self.confidence_history):.2f}")
        
        # Class distribution
        total_predictions = sum(self.class_counts.values())
        if total_predictions > 0:
            print("\nClass Distribution:")
            for class_name in sorted(self.class_counts.keys(),
                                    key=lambda x: self.class_counts[x], reverse=True):
                count = self.class_counts[class_name]
                percentage = (count / total_predictions) * 100
                bar_length = int(percentage / 2)
                bar = "█" * bar_length
                print(f"  {class_name:15}: {percentage:5.1f}% {bar}")
        
        print("=" * 50 + "\n")
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.frame_count = 0
        self.class_counts.clear()
        self.fps_history.clear()
        self.confidence_history.clear()
        self.start_time = time.time()
        print("Statistics reset")
    
    def run(self):
        """Run the live demo."""
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + " " * 12 + "BODY LANGUAGE TFLITE DEMO" + " " * 21 + "║")
        print("╚" + "═" * 58 + "╝\n")
        
        print("Using TensorFlow Lite model for body language classification")
        print(f"Model: {os.path.basename(self.model_path)}\n")
        
        print("Initializing demo...")
        print("-" * 50)
        
        # Load model
        if not self.load_model():
            return
        
        # Initialize face detector
        if not self.initialize_face_detector():
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
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Run inference
                predicted_class, confidence, probabilities = self.predict(frame)
                
                # Update statistics
                self.class_counts[predicted_class] += 1
                self.confidence_history.append(confidence)
                
                # Draw results
                frame = self.draw_results(frame, predicted_class, confidence, probabilities)
                frame = self.draw_info_panel(frame)
                
                # Update frame count and FPS
                self.frame_count += 1
                frame_time = time.time() - frame_start
                if frame_time > 0:
                    self.fps_history.append(1.0 / frame_time)
                
                # Display
                cv2.imshow('Body Language TFLite Demo', frame)
                
                # Handle keys
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
    
    parser = argparse.ArgumentParser(description='Body Language TFLite Model Demo')
    parser.add_argument('--model', type=str,
                       default='../models/body_language.tflite',
                       help='Path to TFLite model file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Resolve model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model)
    
    # Create and run demo
    demo = BodyLanguageTFLiteDemo(model_path=model_path, camera_id=args.camera)
    demo.run()


if __name__ == '__main__':
    main()
