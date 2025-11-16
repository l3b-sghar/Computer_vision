"""
Gender Classification on Video using Hugging Face Transformer
Model: rizvandwiki/gender-classification

Processes video frames and detects gender using the transformer model.
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


class GenderClassifier:
    """Gender classification using Hugging Face transformer model."""
    
    def __init__(self, video_path, model_name="rizvandwiki/gender-classification"):
        """
        Initialize the gender classifier.
        
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
        
        # Statistics
        self.frame_count = 0
        self.gender_counts = {"male": 0, "female": 0}
        self.confidence_scores = []
        
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
            print(f"  Input size: {self.processor.size}")
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
    
    def predict_gender(self, image):
        """
        Predict gender from image.
        
        Args:
            image: PIL Image or numpy array (BGR)
        
        Returns:
            (label, confidence)
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
        label = self.model.config.id2label[predicted_class]
        
        return label, confidence
    
    def process_video(self, skip_frames=5, display=True):
        """
        Process video and classify gender in frames.
        
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
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("\nEnd of video")
                    break
                
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - fps_time >= 1.0:
                    fps_calc = self.frame_count / (current_time - fps_time)
                    fps_time = current_time
                    self.frame_count = 0
                
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
                                # Predict gender
                                label, confidence = self.predict_gender(face_crop)
                                
                                # Store results
                                if label.lower() in ["male", "female"]:
                                    self.gender_counts[label.lower()] += 1
                                    self.confidence_scores.append(confidence)
                                
                                # Draw results
                                color = (0, 255, 0) if label.lower() == "male" else (255, 0, 255)
                                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                
                                text = f"{label.upper()} ({confidence:.2f})"
                                cv2.putText(frame, text, (x, y - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        # No face detected, use full frame
                        label, confidence = self.predict_gender(frame)
                        
                        if label.lower() in ["male", "female"]:
                            self.gender_counts[label.lower()] += 1
                            self.confidence_scores.append(confidence)
                        
                        # Draw on top corner
                        color = (0, 255, 0) if label.lower() == "male" else (255, 0, 255)
                        text = f"{label.upper()} ({confidence:.2f})"
                        cv2.putText(frame, text, (10, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                
                # Draw info panel
                self.draw_info_panel(frame, fps_calc)
                
                # Display
                if display:
                    cv2.imshow('Gender Classification - Press Q to quit', frame)
                    
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
        panel_height = 120
        
        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Gender counts
        total = sum(self.gender_counts.values())
        if total > 0:
            male_pct = (self.gender_counts["male"] / total) * 100
            female_pct = (self.gender_counts["female"] / total) * 100
            
            cv2.putText(frame, f"Male: {self.gender_counts['male']} ({male_pct:.1f}%)", 
                       (10, h - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Female: {self.gender_counts['female']} ({female_pct:.1f}%)", 
                       (10, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Model name
        cv2.putText(frame, f"Model: {self.model_name.split('/')[-1]}", 
                   (w - 350, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def print_statistics(self, elapsed_time):
        """Print final statistics."""
        print("\n" + "=" * 60)
        print("GENDER CLASSIFICATION RESULTS")
        print("=" * 60)
        print(f"Total samples: {sum(self.gender_counts.values())}")
        print(f"Processing time: {elapsed_time:.2f}s")
        
        total = sum(self.gender_counts.values())
        if total > 0:
            print("\nGender Distribution:")
            for gender, count in self.gender_counts.items():
                percentage = (count / total) * 100
                bar = "█" * int(percentage / 2)
                print(f"  {gender.capitalize():10s}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        if self.confidence_scores:
            avg_conf = np.mean(self.confidence_scores)
            print(f"\nAverage confidence: {avg_conf:.2%}")
        
        print("=" * 60)


def main():
    """Main entry point."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "GENDER CLASSIFICATION" + " " * 25 + "║")
    print("║" + " " * 8 + "Hugging Face Transformer Model" + " " * 19 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Configuration
    video_path = r"../data_manipulator/Data_sample_Time_processing_&_Emotion_Detection/sample_cam2.mp4"
    model_name = "rizvandwiki/gender-classification"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        print("Please update the video_path in the script")
        return
    
    # Initialize classifier
    classifier = GenderClassifier(video_path, model_name)
    
    # Load model
    if not classifier.load_model():
        return
    
    # Load face detector
    classifier.load_face_detector()
    
    # Process video
    classifier.process_video(skip_frames=5, display=True)


if __name__ == "__main__":
    main()
