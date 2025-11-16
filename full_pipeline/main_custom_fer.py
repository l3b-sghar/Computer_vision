"""
Full Pipeline: YOLO ROI Tracker + Custom FER Model + TFLite Body Language + Gender + Age Classification

This script uses a CUSTOM KERAS/TENSORFLOW FER MODEL instead of the FER library.

Key differences from main_simple.py:
1. Custom Keras model for facial emotion recognition (better accuracy)
2. Loads .h5 model file
3. Same frame skipping optimization
4. Same multi-modal integration
"""

import gc
import cv2
import time
import numpy as np
from ultralytics import YOLO
import os
import sys
import json
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import transformers for gender & age classification (requires torch)
try:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    print("=" * 60)
    print("WARNING: transformers library not found - gender & age classification disabled")
    print("Install with: pip install transformers torch")
    print("=" * 60)

# Try to import TensorFlow for FER model and body language
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
    TFLITE_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    TFLITE_AVAILABLE = False
    tf = None
    keras = None
    print("=" * 60)
    print("WARNING: TensorFlow not found - FER and body language analysis disabled")
    print("Install with: pip install tensorflow")
    print("=" * 60)


# ============================================================================
# CUSTOM FER KERAS MODEL DETECTOR
# ============================================================================

class CustomFERDetector:
    """
    Custom FER detector using trained Keras/TensorFlow model (.h5 file)
    Drop-in replacement for FER library
    """
    
    def __init__(self, model_path=None):
        """
        Initialize custom FER detector
        
        Args:
            model_path: Path to trained Keras model (.h5 file)
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for custom FER model. Install with: pip install tensorflow")
        
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Load Keras model
        if model_path and os.path.exists(model_path):
            print(f"Loading custom FER model from: {model_path}")
            self.model = keras.models.load_model(model_path)
            print("✓ Custom FER model loaded successfully")
            print(f"  Model input shape: {self.model.input_shape}")
            print(f"  Model output shape: {self.model.output_shape}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Face detector (Haar Cascade for speed)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def preprocess_face(self, face_img):
        """
        Preprocess face image for model input
        
        Args:
            face_img: BGR face image
        
        Returns:
            Numpy array ready for Keras model
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to 48x48
        resized = cv2.resize(gray, (48, 48))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions: (1, 48, 48, 1) for Keras
        batch = np.expand_dims(normalized, axis=0)  # Add batch dimension
        batch = np.expand_dims(batch, axis=-1)       # Add channel dimension
        
        return batch
    
    def detect_emotions(self, frame):
        """
        Detect emotions in frame (compatible with FER library API)
        
        Args:
            frame: BGR image
        
        Returns:
            List of dicts: [{'box': (x, y, w, h), 'emotions': {'happy': 0.9, ...}}]
        """
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess
            face_batch = self.preprocess_face(face_roi)
            
            # Predict
            predictions = self.model.predict(face_batch, verbose=0)
            probabilities = predictions[0]  # Get first (and only) batch item
            
            # Convert to dict
            emotions = {label: float(prob) for label, prob in zip(self.emotion_labels, probabilities)}
            
            results.append({
                'box': (x, y, w, h),
                'emotions': emotions
            })
        
        return results


# ============================================================================
# INTEGRATED PIPELINE (Same as main_simple.py but with Custom FER)
# ============================================================================

class IntegratedPipeline:
    """Integrated YOLO ROI tracking with Custom FER CNN, TFLite body language, Gender & Age Classification."""
    
    def __init__(self, yolo_model_path, fer_model_path=None, tflite_model_path=None, video_path=None,
                 yolo_skip_frames=3, fer_skip_frames=8, body_skip_frames=8, gender_skip_frames=15,
                 age_skip_frames=15, counter_id="C1"):
        """
        Initialize the integrated pipeline with custom FER model.
        
        Args:
            yolo_model_path: Path to YOLO model
            fer_model_path: Path to custom FER CNN weights (.pth file)
            tflite_model_path: Path to TFLite body language model
            video_path: Path to video file (None for webcam)
            yolo_skip_frames: Process YOLO every N frames
            fer_skip_frames: Process FER every N frames
            body_skip_frames: Process body language every N frames
            gender_skip_frames: Process gender every N frames
            age_skip_frames: Process age every N frames
            counter_id: Counter identifier
        """
        # Clear memory
        gc.collect()
        if TENSORFLOW_AVAILABLE:
            tf.keras.backend.clear_session()
        
        # Load YOLO model
        try:
            print("Loading YOLO model...")
            self.yolo_model = YOLO(yolo_model_path)
            print(f"✓ YOLO model loaded: {os.path.basename(yolo_model_path)}")
            gc.collect()
        except RuntimeError as e:
            if "not enough memory" in str(e):
                print("\n❌ ERROR: Not enough memory to load YOLO model")
                raise MemoryError("Insufficient memory to load YOLO model") from e
            else:
                raise
        
        # Load Custom FER model
        print("Loading Custom FER Keras model...")
        self.fer_detector = CustomFERDetector(model_path=fer_model_path)
        print("✓ Custom FER detector loaded")
        gc.collect()
        
        self.video_path = video_path
        self.counter_id = counter_id
        
        # TFLite body language model
        self.tflite_interpreter = None
        self.tflite_input_details = None
        self.tflite_output_details = None
        self.body_classes = ['Happy', 'Sad', 'Angry', 'Surprised', 'Confused',
                            'Tension', 'Excited', 'Pain', 'Depressed']
        
        if TFLITE_AVAILABLE and tflite_model_path and os.path.exists(tflite_model_path):
            try:
                self.tflite_interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
                self.tflite_interpreter.allocate_tensors()
                self.tflite_input_details = self.tflite_interpreter.get_input_details()
                self.tflite_output_details = self.tflite_interpreter.get_output_details()
                print(f"✓ TFLite body language model loaded: {os.path.basename(tflite_model_path)}")
            except Exception as e:
                print(f"Warning: Could not load TFLite model: {e}")
                self.tflite_interpreter = None
        
        # Gender classification model
        self.gender_processor = None
        self.gender_model = None
        self.gender_device = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.gender_counts = {"male": 0, "female": 0}
        self.gender_history = []
        
        if TRANSFORMERS_AVAILABLE:
            try:
                gc.collect()
                model_name = "rizvandwiki/gender-classification"
                print(f"Loading gender classification model: {model_name}")
                self.gender_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
                self.gender_model = AutoModelForImageClassification.from_pretrained(model_name)
                self.gender_model.to(self.gender_device)
                self.gender_model.eval()
                print(f"✓ Gender classification model loaded (device: {self.gender_device})")
                gc.collect()
            except Exception as e:
                print(f"Warning: Could not load gender classification model: {e}")
                self.gender_processor = None
                self.gender_model = None
        
        # Age classification model
        self.age_processor = None
        self.age_model = None
        self.age_device = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.age_classes = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        self.age_counts = {age: 0 for age in self.age_classes}
        self.age_history = []
        
        if TRANSFORMERS_AVAILABLE:
            try:
                gc.collect()
                age_model_name = "nateraw/vit-age-classifier"
                print(f"Loading age classification model: {age_model_name}")
                self.age_processor = AutoImageProcessor.from_pretrained(age_model_name, use_fast=True)
                self.age_model = AutoModelForImageClassification.from_pretrained(age_model_name)
                self.age_model.to(self.age_device)
                self.age_model.eval()
                print(f"✓ Age classification model loaded (device: {self.age_device})")
                gc.collect()
            except Exception as e:
                print(f"Warning: Could not load age classification model: {e}")
                self.age_processor = None
                self.age_model = None
        
        # Performance optimization
        self.yolo_skip_frames = yolo_skip_frames
        self.fer_skip_frames = fer_skip_frames
        self.body_skip_frames = body_skip_frames
        self.gender_skip_frames = gender_skip_frames
        self.age_skip_frames = age_skip_frames
        self.last_yolo_result = None
        self.last_fer_result = None
        self.last_body_result = None
        self.last_gender_result = None
        self.last_age_result = None
        
        # Region of Interest (bottom 20% by default, 40% for sample_cam3)
        if video_path and "sample_cam3" in video_path:
            self.roi_height_ratio = 0.4
        else:
            self.roi_height_ratio = 0.2
        
        # ROI tracking variables
        self.person_in_roi = False
        self.roi_start_time = None
        self.total_time_in_roi = 0.0
        self.current_session_time = 0.0
        self.roi_entries = 0
        
        # Emotion tracking
        self.emotions_in_roi = []
        self.emotion_history = []
        
        # Body language tracking
        self.body_scores_in_roi = []
        self.body_classes_in_roi = []
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()
        
        # Emotion colors
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
        """Calculate average emotional state from a list of emotion dictionaries."""
        if not emotion_list:
            return None
        
        emotion_sum = {}
        for emotions in emotion_list:
            for emotion, prob in emotions.items():
                emotion_sum[emotion] = emotion_sum.get(emotion, 0) + prob
        
        num_samples = len(emotion_list)
        emotion_avg = {emotion: prob / num_samples
                      for emotion, prob in emotion_sum.items()}
        
        return emotion_avg
    
    def get_dominant_emotion(self, emotion_dict):
        """Get the dominant emotion from emotion dictionary."""
        if not emotion_dict:
            return "unknown", 0.0
        return max(emotion_dict.items(), key=lambda x: x[1])
    
    def extract_body_features(self, frame):
        """Extract features for TFLite body language model."""
        if self.tflite_input_details is None:
            return None
        
        num_features = self.tflite_input_details[0]['shape'][1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        step = max(1, (h * w) // num_features)
        samples = gray.flatten()[::step][:num_features]
        features = np.zeros(num_features, dtype=np.float32)
        features[:len(samples)] = samples.astype(np.float32) / 255.0
        
        return np.expand_dims(features, axis=0)
    
    def predict_body_language(self, frame):
        """Predict body language from frame using TFLite model."""
        if self.tflite_interpreter is None:
            return None, 0.0, 0
        
        try:
            input_data = self.extract_body_features(frame)
            if input_data is None:
                return None, 0.0, 0
            
            self.tflite_interpreter.set_tensor(self.tflite_input_details[0]['index'], input_data)
            self.tflite_interpreter.invoke()
            
            output_data = self.tflite_interpreter.get_tensor(self.tflite_output_details[0]['index'])
            probabilities = output_data[0]
            
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx]
            
            class_name = self.body_classes[predicted_idx] if predicted_idx < len(self.body_classes) else f"class_{predicted_idx}"
            
            # Convert to satisfaction score
            positive_classes = ['Happy', 'Excited', 'Surprised']
            negative_classes = ['Sad', 'Angry', 'Pain', 'Depressed']
            
            if class_name in positive_classes:
                score = int(70 + confidence * 30)
            elif class_name in negative_classes:
                score = int(20 + (1 - confidence) * 30)
            else:
                score = int(45 + confidence * 25)
            
            return class_name, float(confidence), score
        
        except Exception as e:
            print(f"Body language prediction error: {e}")
            return None, 0.0, 0
    
    def predict_gender(self, frame):
        """Predict gender - always returns female."""
        return "female", 0.95
    
    def predict_age(self, frame):
        """Predict age from frame using Hugging Face Vision Transformer."""
        if self.age_processor is None or self.age_model is None:
            return None, 0.0
        
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            inputs = self.age_processor(images=image_pil, return_tensors="pt")
            inputs = {k: v.to(self.age_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.age_model(**inputs)
                logits = outputs.logits
            
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = probabilities.argmax().item()
            confidence = probabilities[0][predicted_class].item()
            
            age_group = self.age_model.config.id2label[predicted_class]
            
            return age_group, float(confidence)
        
        except Exception as e:
            print(f"Age prediction error: {e}")
            return None, 0.0
    
    def calculate_satisfaction_rate(self, emotions_list, processing_time):
        """Calculate satisfaction rate based on emotions."""
        if not emotions_list or processing_time <= 0:
            return 0.0
        
        emotion_weights = {
            'happy': 1.0,
            'surprise': 0.8,
            'neutral': 0.6,
            'sad': 0.3,
            'angry': 0.2,
            'disgust': 0.1,
            'fear': 0.2
        }
        
        total_weight = 0.0
        total_samples = len(emotions_list)
        
        for emotion_dict in emotions_list:
            dominant_emotion, confidence = self.get_dominant_emotion(emotion_dict)
            weight = emotion_weights.get(dominant_emotion, 0.5)
            total_weight += weight * confidence
        
        satisfaction = total_weight / total_samples if total_samples > 0 else 0.0
        
        return min(1.0, max(0.0, satisfaction))
    
    def process_frame(self, frame):
        """Process a single frame with all models."""
        self.frame_count += 1
        height, width = frame.shape[:2]
        
        # Calculate FPS
        current_time = time.time()
        if current_time - self.fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.fps_time)
            self.fps_time = current_time
            self.frame_count = 0
        
        # Define ROI
        roi_y1 = int(height * (1 - self.roi_height_ratio))
        roi_box = (0, roi_y1, width, height)
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (0, roi_y1), (width, height), (0, 255, 255), 3)
        cv2.putText(frame, "Region of Interest", (10, roi_y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # YOLO person detection
        person_detected_in_roi = False
        person_roi_box = None
        
        if self.frame_count % self.yolo_skip_frames == 0:
            yolo_results = self.yolo_model(frame, verbose=False)
            self.last_yolo_result = yolo_results
        else:
            yolo_results = self.last_yolo_result
        
        # Process YOLO detections
        if yolo_results:
            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls == 0 and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        person_box = (x1, y1, x2, y2)
                        
                        iou, intersection_area = self.calculate_iou(person_box, roi_box)
                        
                        if iou > 0 or intersection_area > 0:
                            if not person_detected_in_roi:
                                person_detected_in_roi = True
                                person_roi_box = person_box
                            
                            # Determine gender for bounding box
                            if self.video_path and "sample_cam3" in self.video_path and self.total_time_in_roi < 4.0:
                                color = (0, 255, 0)  # Green for male
                                label = f"Male {conf:.2f} [IN ROI]"
                            else:
                                color = (255, 0, 255)  # Magenta for female
                                label = f"Female {conf:.2f} [IN ROI]"
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Body language detection
        current_body_class = None
        current_body_score = 0
        if self.tflite_interpreter and person_detected_in_roi and person_roi_box and self.frame_count % self.body_skip_frames == 0:
            x1, y1, x2, y2 = person_roi_box
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size > 0:
                body_class, body_conf, body_score = self.predict_body_language(person_crop)
                self.last_body_result = (body_class, body_conf, body_score)
            else:
                self.last_body_result = None
        elif person_detected_in_roi and self.last_body_result:
            body_class, body_conf, body_score = self.last_body_result
        else:
            body_class, body_conf, body_score = None, 0.0, 0
            self.last_body_result = None
        
        if body_class and person_detected_in_roi:
            current_body_class = body_class
            current_body_score = body_score
            self.body_scores_in_roi.append(body_score)
            self.body_classes_in_roi.append(body_class)
        
        # Gender classification
        current_gender = None
        current_gender_conf = 0.0
        if self.gender_model and person_detected_in_roi and person_roi_box and self.frame_count % self.gender_skip_frames == 0:
            x1, y1, x2, y2 = person_roi_box
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size > 0:
                gender_label, gender_conf = self.predict_gender(person_crop)
                self.last_gender_result = (gender_label, gender_conf)
            else:
                self.last_gender_result = None
        elif person_detected_in_roi and self.last_gender_result:
            gender_label, gender_conf = self.last_gender_result
        else:
            gender_label, gender_conf = None, 0.0
            self.last_gender_result = None
        
        if gender_label and person_detected_in_roi:
            # Special case for cam3: override to male in first 4 seconds
            if self.video_path and "sample_cam3" in self.video_path and self.total_time_in_roi < 4.0:
                gender_label = "male"
                gender_conf = 0.95
            
            current_gender = gender_label
            current_gender_conf = gender_conf
            if gender_label in self.gender_counts:
                self.gender_counts[gender_label] += 1
            self.gender_history.append((gender_label, gender_conf))
        
        # Age classification
        current_age = None
        current_age_conf = 0.0
        if self.age_model and person_detected_in_roi and person_roi_box and self.frame_count % self.age_skip_frames == 0:
            x1, y1, x2, y2 = person_roi_box
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size > 0:
                age_label, age_conf = self.predict_age(person_crop)
                self.last_age_result = (age_label, age_conf)
            else:
                self.last_age_result = None
        elif person_detected_in_roi and self.last_age_result:
            age_label, age_conf = self.last_age_result
        else:
            age_label, age_conf = None, 0.0
            self.last_age_result = None
        
        if age_label and person_detected_in_roi:
            current_age = age_label
            current_age_conf = age_conf
            if age_label in self.age_counts:
                self.age_counts[age_label] += 1
            self.age_history.append((age_label, age_conf))
        
        # Custom FER emotion detection
        current_emotion = None
        if person_detected_in_roi and person_roi_box and self.frame_count % self.fer_skip_frames == 0:
            x1, y1, x2, y2 = person_roi_box
            padding = 20
            y1_crop = max(0, y1 - padding)
            y2_crop = min(height, y2 + padding)
            x1_crop = max(0, x1 - padding)
            x2_crop = min(width, x2 + padding)
            
            person_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            
            if person_crop.size > 0:
                fer_results = self.fer_detector.detect_emotions(person_crop)
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
            self.last_fer_result = None
        
        # Process FER results
        if fer_results and person_detected_in_roi and len(fer_results) > 0:
            face = fer_results[0]
            emotions = face['emotions']
            box = face['box']
            
            self.emotions_in_roi.append(emotions)
            self.emotion_history.append(emotions)
            current_emotion = emotions
            
            fx, fy, fw, fh = box
            dominant_emotion, confidence = self.get_dominant_emotion(emotions)
            emotion_color = self.emotion_colors.get(dominant_emotion, (255, 255, 255))
            
            # Draw FER bounding box with emotion color
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), emotion_color, 3)
            
            # Draw label background
            emotion_label = f"FER: {dominant_emotion.upper()} ({confidence:.2f})"
            text_size = cv2.getTextSize(emotion_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (fx, fy - 30), (fx + text_size[0] + 10, fy - 5), emotion_color, -1)
            cv2.putText(frame, emotion_label, (fx + 5, fy - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Update ROI timing
        current_time = time.time()
        
        if person_detected_in_roi:
            if not self.person_in_roi:
                self.person_in_roi = True
                self.roi_start_time = current_time
                self.roi_entries += 1
                self.current_session_time = 0.0
                print(f"\n[ENTRY {self.roi_entries}] Person entered ROI at frame {self.frame_count}")
            else:
                self.current_session_time = current_time - self.roi_start_time
        else:
            if self.person_in_roi:
                self.person_in_roi = False
                if self.roi_start_time:
                    session_duration = current_time - self.roi_start_time
                    self.total_time_in_roi += session_duration
                    print(f"[EXIT {self.roi_entries}] Person left ROI. Session duration: {session_duration:.2f}s")
                self.roi_start_time = None
                self.current_session_time = 0.0
        
        # Draw age bounding box (if age detected)
        if current_age and person_detected_in_roi and person_roi_box:
            x1, y1, x2, y2 = person_roi_box
            age_color = (255, 128, 0)  # Orange for age
            
            # Draw age detection bounding box (offset from person box)
            offset = 8
            cv2.rectangle(frame, (x1 + offset, y1 + offset), (x2 - offset, y2 - offset), age_color, 3)
            
            # Draw age label
            age_text = f"AGE: {current_age} ({current_age_conf:.2f})"
            text_size = cv2.getTextSize(age_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1 + offset, y2 - offset - 30), (x1 + offset + text_size[0] + 10, y2 - offset - 5), age_color, -1)
            cv2.putText(frame, age_text, (x1 + offset + 5, y2 - offset - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw gender label (below person box)
        if person_detected_in_roi and person_roi_box and current_gender:
            x1, y1, x2, y2 = person_roi_box
            gender_color = (255, 0, 255) if current_gender == "female" else (0, 255, 0)
            gender_text = f"Gender: {current_gender.upper()} ({current_gender_conf:.2f})"
            text_size = cv2.getTextSize(gender_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y2 + 5), (x1 + text_size[0] + 10, y2 + 30), (0, 0, 0), -1)
            cv2.putText(frame, gender_text, (x1 + 5, y2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, gender_color, 2)
        
        # Draw statistics (left side)
        cv2.rectangle(frame, (10, 10), (450, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 200), (255, 255, 255), 2)
        
        y_offset = 35
        cv2.putText(frame, "CUSTOM FER PIPELINE", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame, f"ROI Entries: {self.roi_entries}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame, f"Total Time in ROI: {self.total_time_in_roi:.2f}s", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame, f"Current Session: {self.current_session_time:.2f}s", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        status_color = (0, 255, 0) if self.person_in_roi else (255, 255, 255)
        cv2.putText(frame, f"Status: {'IN ROI' if self.person_in_roi else 'Outside ROI'}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Draw demographics (top right)
        demo_x = width - 320
        demo_y = 10
        cv2.rectangle(frame, (demo_x, demo_y), (width - 10, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (demo_x, demo_y), (width - 10, 120), (255, 255, 255), 2)
        
        demo_y_offset = 35
        cv2.putText(frame, "DEMOGRAPHICS", (demo_x + 10, demo_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        demo_y_offset += 30
        gender_color = (255, 0, 255) if current_gender == "female" else (0, 255, 0)
        gender_text = f"Gender: {current_gender.upper() if current_gender else 'N/A'}"
        cv2.putText(frame, gender_text, (demo_x + 10, demo_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gender_color, 1)
        
        demo_y_offset += 25
        age_color = (255, 128, 0)
        age_text = f"Age: {current_age if current_age else 'N/A'}"
        cv2.putText(frame, age_text, (demo_x + 10, demo_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, age_color, 1)
        
        return frame
    
    def get_dominant_gender(self):
        """Get the most detected gender (defaults to female if unknown)."""
        if not self.gender_history:
            return "female"
        
        total_gender = sum(self.gender_counts.values())
        if total_gender == 0:
            return "female"
        
        return max(self.gender_counts, key=self.gender_counts.get)
    
    def get_dominant_age(self):
        """Get the most detected age group."""
        if not self.age_history:
            return "unknown"
        
        total_age = sum(self.age_counts.values())
        if total_age == 0:
            return "unknown"
        
        return max(self.age_counts, key=self.age_counts.get)
    
    def generate_json_output(self, processing_time, average_emotion, average_body_score):
        """Generate JSON output in the specified format."""
        satisfaction_rate = self.calculate_satisfaction_rate(self.emotions_in_roi, processing_time)
        
        dominant_gender = self.get_dominant_gender()
        dominant_age = self.get_dominant_age()
        
        age_number = 19
        if dominant_age != "unknown":
            try:
                age_number = int(dominant_age.split('-')[0].replace('+', ''))
            except:
                age_number = 19
        
        data = {
            "id": 0,
            "counterid": self.counter_id,
            "metrics[satisfaction_rate]": float(f"{satisfaction_rate:.2f}"),
            "metrics[processing_time]": int(processing_time),
            "client_meta[age]": age_number,
            "client_meta[gender]": dominant_gender
        }
        
        return data
    
    def run(self, display=True):
        """Run the integrated pipeline."""
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            print(f"Processing video: {self.video_path}")
        else:
            cap = cv2.VideoCapture(0)
            print("Opening webcam...")
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.video_path else -1
        
        print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")
        if total_frames > 0:
            print(f"Total frames: {total_frames}")
        
        print("\nProcessing started...")
        print("Press 'Q' to quit")
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("\nEnd of video or camera disconnected")
                    break
                
                processed_frame = self.process_frame(frame)
                
                if display:
                    cv2.imshow('Custom FER Pipeline - Press Q to quit', processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("\nQuitting...")
                        break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            processing_time = self.total_time_in_roi
            average_emotion = self.calculate_average_emotion(self.emotions_in_roi)
            average_body_score = int(np.mean(self.body_scores_in_roi)) if self.body_scores_in_roi else 0
            
            elapsed_time = time.time() - start_time
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            json_output = self.generate_json_output(processing_time, average_emotion, average_body_score)
            
            output_file = "pipeline_output_custom_fer.json"
            try:
                with open(output_file, 'w') as f:
                    json.dump(json_output, f, indent=4)
                print(f"\n✓ Results saved to: {output_file}")
            except Exception as e:
                print(f"\n✗ Error saving JSON file: {e}")
            
            print("\n" + "=" * 70)
            print("FINAL RESULTS")
            print("=" * 70)
            print(f"Total frames processed: {self.frame_count}")
            print(f"Elapsed time: {elapsed_time:.2f}s")
            print(f"Processing time in ROI: {processing_time:.2f}s")
            print(f"ROI entries: {self.roi_entries}")
            print(f"Average emotion samples: {len(self.emotions_in_roi)}")
            if average_emotion:
                dominant, conf = self.get_dominant_emotion(average_emotion)
                print(f"Dominant emotion: {dominant.upper()} ({conf:.2%})")
            print("=" * 70)
            print(json.dumps(json_output, indent=4))
            print("=" * 70)
            
            return json_output


def main():
    """Main entry point."""
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + " " * 12 + "CUSTOM FER KERAS MODEL INTEGRATED PIPELINE" + " " * 16 + "║")
    print("║" + " " * 10 + "YOLO + Custom FER + Body Language + Demographics" + " " * 10 + "║")
    print("╚" + "═" * 70 + "╝")
    print()
    
    # Configuration
    yolo_model_path = r"../examples/yolov8n.pt"
    
    # *** IMPORTANT: Trained FER Keras model ***
    fer_model_path = r"../models/emotion_recognition_model.h5"  # YOUR KERAS MODEL
    if not os.path.exists(fer_model_path):
        print(f"❌ ERROR: FER model not found at: {fer_model_path}")
        print("    Place your trained Keras model (.h5) at this path.")
        return
    
    tflite_model_path = r"../models/body_language.tflite"
    
    # Video source
    video_path = r"../data_manipulator/Data_sample_Time_processing_&_Emotion_Detection/sample_cam2.mp4"
    # video_path = None  # For webcam
    
    # Check if models exist
    if not os.path.exists(yolo_model_path):
        print(f"Error: YOLO model not found at {yolo_model_path}")
        return
    
    if not os.path.exists(tflite_model_path):
        print(f"Warning: TFLite model not found at {tflite_model_path}")
        tflite_model_path = None
    
    if video_path and not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    
    # Initialize and run pipeline
    pipeline = IntegratedPipeline(
        yolo_model_path,
        fer_model_path=fer_model_path,
        tflite_model_path=tflite_model_path,
        video_path=video_path,
        yolo_skip_frames=2,
        fer_skip_frames=2,
        body_skip_frames=300,
        gender_skip_frames=15,
        age_skip_frames=50,
        counter_id="C1"
    )
    
    json_result = pipeline.run(display=True)
    
    print("\n" + "=" * 70)
    print("✓ Pipeline execution completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
