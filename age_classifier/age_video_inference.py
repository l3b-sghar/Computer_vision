"""
Age Classification on Video using Hugging Face Transformer
Model: nateraw/vit-age-classifier

Processes video frames and detects age using the Vision Transformer model.
Age classes: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
"""

import cv2
import numpy as np
import time
from PIL import Image
import os
import sys

# Try to import transformers
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("=" * 60)
    print("ERROR: transformers library not found")
    print("Please install: pip install transformers torch pillow")
    print("=" * 60)
    sys.exit(1)


class AgeClassifier:
    """Age classification using Hugging Face Vision Transformer model."""
    
    def __init__(self, video_path, model_name="nateraw/vit-age-classifier"):
        """
        Initialize the age classifier.
        
        Args:
            video_path: Path to video file
            model_name: Hugging Face model name
        """
        self.video_path = video_path
        self.model_name = model_name
        
        # Model and processor
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Age groups from the model
        self.age_classes = ['0-2', '3-9', '10-19', '20-29', '30-39', 
                           '40-49', '50-59', '60-69', '70+']
        
        # Statistics
        self.frame_count = 0
        self.age_counts = {age: 0 for age in self.age_classes}
        self.confidence_scores = []
        self.age_history = []
        
        # Face detection for better results
        self.face_cascade = None
        
    def load_model(self):
        """Load the Hugging Face model."""
        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}")
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Model loaded successfully")
            print(f"  Age classes: {', '.join(self.age_classes)}")
            return True
        
        except Exception as e:
            print(f"ERROR loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_face_detector(self):
        """Load face detector for better cropping."""
        print("Loading face detector...")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("Warning: Could not load face cascade, using full frame")
            return False
        
        print("✓ Face detector loaded")
        return True
    
    def detect_faces(self, frame):
        """
        Detect faces in frame.
        
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    
    def predict_age(self, image):
        """
        Predict age from image.
        
        Args:
            image: PIL Image or numpy array (BGR)
        
        Returns:
            (age_group, confidence)
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get prediction
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = probabilities.argmax().item()
        confidence = probabilities[0][predicted_class].item()
        
        # Map to label
        age_group = self.model.config.id2label[predicted_class]
        
        return age_group, confidence
    
    def get_age_color(self, age_group):
        """Get color for age group visualization."""
        age_colors = {
            '0-2': (255, 200, 200),    # Light pink
            '3-9': (255, 150, 150),    # Pink
            '10-19': (200, 255, 200),  # Light green
            '20-29': (100, 255, 100),  # Green
            '30-39': (100, 200, 255),  # Light blue
            '40-49': (100, 150, 255),  # Blue
            '50-59': (200, 100, 255),  # Purple
            '60-69': (255, 100, 200),  # Magenta
            '70+': (255, 100, 100)     # Red
        }
        return age_colors.get(age_group, (255, 255, 255))
    
    def process_video(self, skip_frames=5, display=True):
        """
        Process video and classify age in frames.
        
        Args:
            skip_frames: Process every N frames for performance
            display: Whether to display video window
        """
        print(f"\nProcessing video: {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"ERROR: Could not open video: {self.video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")
        print(f"Total frames: {total_frames}")
        print(f"Processing every {skip_frames} frames")
        print("\nPress 'Q' to quit\n")
        
        start_time = time.time()
        fps_calc = 0
        fps_time = time.time()
        frame_counter = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("\nEnd of video")
                    break
                
                frame_counter += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - fps_time >= 1.0:
                    fps_calc = frame_counter / (current_time - fps_time)
                    fps_time = current_time
                    frame_counter = 0
                
                # Process every N frames
                if self.frame_count % skip_frames == 0:
                    # Detect faces
                    faces = self.detect_faces(frame)
                    
                    if len(faces) > 0:
                        # Process each face
                        for (x, y, w, h) in faces:
                            # Crop face with padding
                            padding = 20
                            x1 = max(0, x - padding)
                            y1 = max(0, y - padding)
                            x2 = min(frame_width, x + w + padding)
                            y2 = min(frame_height, y + h + padding)
                            
                            face_crop = frame[y1:y2, x1:x2]
                            
                            if face_crop.size > 0:
                                # Predict age
                                age_group, confidence = self.predict_age(face_crop)
                                
                                # Store results
                                if age_group in self.age_counts:
                                    self.age_counts[age_group] += 1
                                    self.confidence_scores.append(confidence)
                                    self.age_history.append((age_group, confidence))
                                
                                # Draw results
                                color = self.get_age_color(age_group)
                                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                
                                text = f"{age_group} ({confidence:.2f})"
                                cv2.putText(frame, text, (x, y - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        # No face detected, use full frame (less accurate)
                        age_group, confidence = self.predict_age(frame)
                        
                        if age_group in self.age_counts:
                            self.age_counts[age_group] += 1
                            self.confidence_scores.append(confidence)
                            self.age_history.append((age_group, confidence))
                        
                        # Draw on top corner
                        color = self.get_age_color(age_group)
                        text = f"Age: {age_group} ({confidence:.2f})"
                        cv2.putText(frame, text, (10, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                
                self.frame_count += 1
                
                # Draw info panel
                self.draw_info_panel(frame, fps_calc)
                
                # Display
                if display:
                    cv2.imshow('Age Classification - Press Q to quit', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("\nQuitting...")
                        break
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            elapsed_time = time.time() - start_time
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            # Print statistics
            self.print_statistics(elapsed_time)
    
    def draw_info_panel(self, frame, fps):
        """Draw information panel on frame."""
        h, w = frame.shape[:2]
        panel_height = 180
        
        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Age distribution (top 3)
        total = sum(self.age_counts.values())
        if total > 0:
            sorted_ages = sorted(self.age_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            y_offset = h - 120
            for i, (age, count) in enumerate(sorted_ages):
                pct = (count / total) * 100
                color = self.get_age_color(age)
                text = f"{age}: {count} ({pct:.1f}%)"
                cv2.putText(frame, text, (10, y_offset + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Model name
        cv2.putText(frame, f"Model: {self.model_name.split('/')[-1]}", 
                   (w - 350, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def get_dominant_age(self):
        """Get the most detected age group."""
        if not self.age_history:
            return "unknown"
        
        total = sum(self.age_counts.values())
        if total == 0:
            return "unknown"
        
        # Return age with highest count
        return max(self.age_counts, key=self.age_counts.get)
    
    def print_statistics(self, elapsed_time):
        """Print final statistics."""
        print("\n" + "=" * 60)
        print("AGE CLASSIFICATION RESULTS")
        print("=" * 60)
        print(f"Total samples: {sum(self.age_counts.values())}")
        print(f"Processing time: {elapsed_time:.2f}s")
        
        total = sum(self.age_counts.values())
        if total > 0:
            print("\nAge Distribution:")
            sorted_ages = sorted(self.age_counts.items(), key=lambda x: x[1], reverse=True)
            for age, count in sorted_ages:
                if count > 0:
                    percentage = (count / total) * 100
                    bar = "█" * int(percentage / 2)
                    print(f"  {age:10s}: {count:4d} ({percentage:5.1f}%) {bar}")
            
            print(f"\nDominant age group: {self.get_dominant_age()}")
        
        if self.confidence_scores:
            avg_conf = np.mean(self.confidence_scores)
            print(f"Average confidence: {avg_conf:.2%}")
        
        print("=" * 60)


def main():
    """Main entry point."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "AGE CLASSIFICATION" + " " * 25 + "║")
    print("║" + " " * 8 + "Hugging Face Vision Transformer" + " " * 19 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Configuration
    video_path = r"../data_manipulator/Data_sample_Time_processing_&_Emotion_Detection/sample_cam1.mp4"
    model_name = "nateraw/vit-age-classifier"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        print("Please update the video_path in the script")
        return
    
    # Initialize classifier
    classifier = AgeClassifier(video_path, model_name)
    
    # Load model
    if not classifier.load_model():
        return
    
    # Load face detector
    classifier.load_face_detector()
    
    # Process video
    classifier.process_video(skip_frames=5, display=True)


if __name__ == "__main__":
    main()
