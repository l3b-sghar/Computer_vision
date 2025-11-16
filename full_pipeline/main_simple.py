"""
Full Pipeline: YOLO ROI Tracker + FER Emotion Detection + TFLite Body Language + Gender Classification

This script combines:
1. YOLO person detection with ROI time tracking
2. FER emotion detection during ROI presence
3. TFLite body language classification (simple, no MediaPipe needed)
4. Gender classification using Hugging Face transformer
5. Returns: JSON format with satisfaction metrics
"""

import gc  # For garbage collection
import cv2
import time
import numpy as np
from ultralytics import YOLO
import os
import sys
import json
from PIL import Image
import requests

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

# Try to import transformers for gender classification
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("=" * 60)
    print("WARNING: transformers library not found - gender classification disabled")
    print("Install with: pip install transformers torch")
    print("=" * 60)


class IntegratedPipeline:
    """Integrated YOLO ROI tracking with FER emotion detection, TFLite body language, and Gender Classification."""
    
    def __init__(self, yolo_model_path, tflite_model_path=None, video_path=None, 
                 yolo_skip_frames=3, fer_skip_frames=8, body_skip_frames=8, gender_skip_frames=15,
                 age_skip_frames=15, counter_id="C1"):
        """
        Initialize the integrated pipeline.
        
        Args:
            yolo_model_path: Path to YOLO model
            tflite_model_path: Path to TFLite body language model
            video_path: Path to video file (None for webcam)
            yolo_skip_frames: Process YOLO every N frames (3 for smooth tracking)
            fer_skip_frames: Process FER every N frames (8 for performance)
            body_skip_frames: Process body language every N frames (8 for performance)
            gender_skip_frames: Process gender every N frames (15 for transformer efficiency)
            age_skip_frames: Process age every N frames (15 for transformer efficiency)
            counter_id: Counter identifier (e.g., "C1", "C2")
        """
        # Clear memory before loading heavy models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load YOLO model with error handling for memory issues
        try:
            print("Loading YOLO model...")
            self.yolo_model = YOLO(yolo_model_path)
            print(f"✓ YOLO model loaded: {os.path.basename(yolo_model_path)}")
            gc.collect()  # Free any temporary memory used during loading
        except RuntimeError as e:
            if "not enough memory" in str(e):
                print("\n❌ ERROR: Not enough memory to load YOLO model")
                print("   Try closing other applications and running again")
                print("   Or restart Python to free up memory")
                raise MemoryError("Insufficient memory to load YOLO model") from e
            else:
                raise
        
        print("Loading FER emotion detector...")
        self.fer_detector = FER(mtcnn=True)  # FER emotion detector
        print("✓ FER detector loaded")
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
                gc.collect()  # Free memory before loading transformer
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
                gc.collect()  # Free memory before loading transformer
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
        self.body_scores_in_roi = []
        self.body_classes_in_roi = []
        
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
    
    def extract_body_features(self, frame):
        """
        Extract features for TFLite body language model.
        Uses pixel sampling as workaround (ideally would use pose landmarks).
        """
        if self.tflite_input_details is None:
            return None
        
        num_features = self.tflite_input_details[0]['shape'][1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Sample pixels across frame
        step = max(1, (h * w) // num_features)
        samples = gray.flatten()[::step][:num_features]
        features = np.zeros(num_features, dtype=np.float32)
        features[:len(samples)] = samples.astype(np.float32) / 255.0
        
        return np.expand_dims(features, axis=0)
    
    def predict_body_language(self, frame):
        """
        Predict body language from frame using TFLite model.
        Returns: (class_name, confidence, score_0_100)
        """
        if self.tflite_interpreter is None:
            return None, 0.0, 0
        
        try:
            # Extract features
            input_data = self.extract_body_features(frame)
            if input_data is None:
                return None, 0.0, 0
            
            # Run inference
            self.tflite_interpreter.set_tensor(self.tflite_input_details[0]['index'], input_data)
            self.tflite_interpreter.invoke()
            
            # Get output
            output_data = self.tflite_interpreter.get_tensor(self.tflite_output_details[0]['index'])
            probabilities = output_data[0]
            
            # Get predicted class
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx]
            
            class_name = self.body_classes[predicted_idx] if predicted_idx < len(self.body_classes) else f"class_{predicted_idx}"
            
            # Convert to 0-100 score (higher probability = higher satisfaction)
            # Positive emotions (Happy, Excited, Surprised) get higher scores
            positive_classes = ['Happy', 'Excited', 'Surprised']
            negative_classes = ['Sad', 'Angry', 'Pain', 'Depressed']
            
            if class_name in positive_classes:
                score = int(70 + confidence * 30)  # 70-100 range
            elif class_name in negative_classes:
                score = int(20 + (1 - confidence) * 30)  # 20-50 range
            else:  # Neutral classes
                score = int(45 + confidence * 25)  # 45-70 range
            
            return class_name, float(confidence), score
        
        except Exception as e:
            print(f"Body language prediction error: {e}")
            return None, 0.0, 0
    
    def analyze_skin_features(self, frame):
        """
        Analyze skin tone features that correlate with gender presentation.
        Returns skin smoothness score and brightness score.
        """
        try:
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            
            # Skin detection mask (focusing on face region)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Calculate skin smoothness using variance (lower variance = smoother)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            skin_region = cv2.bitwise_and(gray, gray, mask=skin_mask)
            
            # Variance in skin region (lower = smoother)
            if np.count_nonzero(skin_mask) > 100:
                variance = np.var(skin_region[skin_mask > 0])
                smoothness = 1.0 / (1.0 + variance / 100.0)  # Normalize
            else:
                smoothness = 0.5
            
            # Calculate average brightness in skin areas
            y_channel = ycrcb[:, :, 0]
            skin_brightness = np.mean(y_channel[skin_mask > 0]) if np.count_nonzero(skin_mask) > 0 else 128
            brightness_score = skin_brightness / 255.0
            
            return smoothness, brightness_score
        
        except Exception as e:
            return 0.5, 0.5  # Neutral values on error
    
    def predict_gender(self, frame):
        """
        Predict gender from frame using transformer with skin feature analysis.
        Returns: (label, confidence) e.g., ("female", 0.95)
        """
        if self.gender_processor is None or self.gender_model is None:
            return None, 0.0
        
        try:
            # Analyze skin features for additional context
            smoothness, brightness = self.analyze_skin_features(frame)
            
            # Simple RGB conversion for model
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # Preprocess with model processor
            inputs = self.gender_processor(images=image_pil, return_tensors="pt")
            inputs = {k: v.to(self.gender_device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.gender_model(**inputs)
                logits = outputs.logits
            
            # Get raw probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # Get model's prediction
            predicted_class = probabilities.argmax()
            base_confidence = probabilities[predicted_class]
            label = self.gender_model.config.id2label[predicted_class].lower()
            
            # Adjust prediction based on skin features
            # Smoother, lighter skin correlates with female presentation
            skin_female_score = (smoothness * 0.6 + brightness * 0.4)
            
            # If skin features suggest female and confidence is not very high
            if skin_female_score > 0.6 and base_confidence < 0.85:
                # Check if we should override prediction
                if label == "male" and skin_female_score > 0.7:
                    # Recalculate with skin bias
                    female_idx = 0 if self.gender_model.config.id2label[0].lower() == "female" else 1
                    male_idx = 1 - female_idx
                    
                    # Boost female probability based on skin features
                    boost_factor = (skin_female_score - 0.5) * 0.3  # Up to 15% boost
                    adjusted_female_prob = min(0.95, probabilities[female_idx] + boost_factor)
                    adjusted_male_prob = 1.0 - adjusted_female_prob
                    
                    # Update prediction if female probability is now higher
                    if adjusted_female_prob > adjusted_male_prob:
                        label = "female"
                        confidence = adjusted_female_prob
                    else:
                        confidence = base_confidence
                else:
                    confidence = base_confidence
            else:
                confidence = base_confidence
            
            return label, float(confidence)
        
        except Exception as e:
            print(f"Gender prediction error: {e}")
            return None, 0.0
    
    def predict_age(self, frame):
        """
        Predict age from frame using Hugging Face Vision Transformer.
        Returns: (age_group, confidence) e.g., ("20-29", 0.85)
        """
        if self.age_processor is None or self.age_model is None:
            return None, 0.0
        
        try:
            # Simple RGB conversion (good lighting = no complex preprocessing needed)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # Preprocess with model processor
            inputs = self.age_processor(images=image_pil, return_tensors="pt")
            inputs = {k: v.to(self.age_device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.age_model(**inputs)
                logits = outputs.logits
            
            # Get prediction with softmax
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = probabilities.argmax().item()
            confidence = probabilities[0][predicted_class].item()
            
            # Map to label
            age_group = self.age_model.config.id2label[predicted_class]
            
            return age_group, float(confidence)
        
        except Exception as e:
            print(f"Age prediction error: {e}")
            return None, 0.0
    
    def calculate_satisfaction_rate(self, emotions_list, processing_time):
        """
        Calculate satisfaction rate based on dominant emotion frequency over processing time.
        
        Formula: (positive_emotion_frequency / total_samples) * emotion_weight
        
        Returns: Float between 0.0 and 1.0
        """
        if not emotions_list or processing_time <= 0:
            return 0.0
        
        # Emotion weights (positive emotions = higher satisfaction)
        emotion_weights = {
            'happy': 1.0,
            'surprise': 0.8,
            'neutral': 0.6,
            'sad': 0.3,
            'angry': 0.2,
            'disgust': 0.1,
            'fear': 0.2
        }
        
        # Calculate weighted satisfaction
        total_weight = 0.0
        total_samples = len(emotions_list)
        
        for emotion_dict in emotions_list:
            dominant_emotion, confidence = self.get_dominant_emotion(emotion_dict)
            weight = emotion_weights.get(dominant_emotion, 0.5)
            total_weight += weight * confidence
        
        # Average satisfaction rate
        satisfaction = total_weight / total_samples if total_samples > 0 else 0.0
        
        return min(1.0, max(0.0, satisfaction))  # Clamp between 0 and 1
    
    def process_frame(self, frame):
        """Process a single frame with YOLO + FER + TFLite (optimized with frame skipping)."""
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
                        
                        # Draw person label at top
                        cv2.putText(frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Run TFLite body language detection ONLY for person in ROI
        current_body_class = None
        current_body_score = 0
        if self.tflite_interpreter and person_detected_in_roi and person_roi_box and self.frame_count % self.body_skip_frames == 0:
            # Crop to person bounding box
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
        
        # Store body language results
        if body_class and person_detected_in_roi:
            current_body_class = body_class
            current_body_score = body_score
            self.body_scores_in_roi.append(body_score)
            self.body_classes_in_roi.append(body_class)
        
        # Run gender classification ONLY for person in ROI (every N frames)
        current_gender = None
        current_gender_conf = 0.0
        if self.gender_model and person_detected_in_roi and person_roi_box and self.frame_count % self.gender_skip_frames == 0:
            # Crop to person bounding box
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
        
        # Store gender results
        if gender_label and person_detected_in_roi:
            current_gender = gender_label
            current_gender_conf = gender_conf
            if gender_label in self.gender_counts:
                self.gender_counts[gender_label] += 1
            self.gender_history.append((gender_label, gender_conf))
        
        # Run age classification ONLY for person in ROI (every N frames)
        current_age = None
        current_age_conf = 0.0
        if self.age_model and person_detected_in_roi and person_roi_box and self.frame_count % self.age_skip_frames == 0:
            # Crop to person bounding box
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
        
        # Store age results
        if age_label and person_detected_in_roi:
            current_age = age_label
            current_age_conf = age_conf
            if age_label in self.age_counts:
                self.age_counts[age_label] += 1
            self.age_history.append((age_label, age_conf))
        
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
        
        # Draw gender and age labels on person bounding box if detected
        if person_detected_in_roi and person_roi_box:
            x1, y1, x2, y2 = person_roi_box
            
            # Draw gender label
            if current_gender:
                gender_color = (255, 0, 255) if current_gender == "female" else (0, 255, 0)
                gender_text = f"Gender: {current_gender.upper()} ({current_gender_conf:.2f})"
                # Draw background rectangle for better visibility
                text_size = cv2.getTextSize(gender_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y2 + 5), (x1 + text_size[0] + 10, y2 + 30), (0, 0, 0), -1)
                cv2.putText(frame, gender_text, (x1 + 5, y2 + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, gender_color, 2)
            
            # Draw age label
            if current_age:
                age_color = (255, 128, 0)  # Orange
                age_text = f"Age: {current_age} ({current_age_conf:.2f})"
                # Draw background rectangle for better visibility
                text_size = cv2.getTextSize(age_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                y_pos = y2 + 35 if current_gender else y2 + 5
                cv2.rectangle(frame, (x1, y_pos), (x1 + text_size[0] + 10, y_pos + 25), (0, 0, 0), -1)
                cv2.putText(frame, age_text, (x1 + 5, y_pos + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, age_color, 2)
        
        # Draw comprehensive statistics
        self.draw_statistics(frame, current_emotion, current_body_class, current_body_score, 
                           current_gender, current_gender_conf, current_age, current_age_conf)
        
        return frame
    
    def draw_statistics(self, frame, current_emotion, current_body_class, current_body_score,
                       current_gender=None, current_gender_conf=0.0, current_age=None, current_age_conf=0.0):
        """Draw statistics overlay on frame."""
        y_offset = 30
        line_height = 25
        
        # Background for statistics (made taller for gender + age info)
        cv2.rectangle(frame, (10, 10), (450, 480), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 480), (255, 255, 255), 2)
        
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
        
        if TFLITE_AVAILABLE and self.tflite_interpreter and current_body_class:
            cv2.putText(frame, f"Current: {current_body_class} ({current_body_score})", 
                       (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            if self.body_scores_in_roi:
                avg_body = int(np.mean(self.body_scores_in_roi))
                cv2.putText(frame, f"Average: {avg_body}", 
                           (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += line_height
        elif not TFLITE_AVAILABLE or not self.tflite_interpreter:
            cv2.putText(frame, "TFLite not available", (20, y_offset),
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
        
        y_offset += line_height + 10
        
        # Gender Classification Statistics
        cv2.putText(frame, "GENDER (IN ROI)", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += line_height
        
        if TRANSFORMERS_AVAILABLE and self.gender_model and current_gender:
            gender_color = (0, 255, 0) if current_gender == "male" else (255, 0, 255)
            cv2.putText(frame, f"Current: {current_gender.upper()} ({current_gender_conf:.2f})", 
                       (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, gender_color, 1)
            y_offset += line_height
            
            # Show gender distribution
            total_gender = sum(self.gender_counts.values())
            if total_gender > 0:
                for gender, count in self.gender_counts.items():
                    pct = (count / total_gender) * 100
                    cv2.putText(frame, f"{gender.capitalize()}: {count} ({pct:.0f}%)", 
                               (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    y_offset += line_height - 5
        elif not TRANSFORMERS_AVAILABLE or not self.gender_model:
            cv2.putText(frame, "Gender model not available", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        y_offset += line_height + 5
        
        # Age Classification Statistics
        cv2.putText(frame, "AGE (IN ROI)", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += line_height
        
        if TRANSFORMERS_AVAILABLE and self.age_model and current_age:
            age_color = (255, 128, 0)  # Orange for age
            cv2.putText(frame, f"Current: {current_age} ({current_age_conf:.2f})", 
                       (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, age_color, 1)
            y_offset += line_height
            
            # Show top 3 age groups
            total_age = sum(self.age_counts.values())
            if total_age > 0:
                sorted_ages = sorted(self.age_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                for age, count in sorted_ages:
                    if count > 0:
                        pct = (count / total_age) * 100
                        cv2.putText(frame, f"{age}: {count} ({pct:.0f}%)", 
                                   (20, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                        y_offset += line_height - 5
        elif not TRANSFORMERS_AVAILABLE or not self.age_model:
            cv2.putText(frame, "Age model not available", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    def get_dominant_gender(self):
        """Get the most detected gender."""
        if not self.gender_history:
            return "unknown"
        
        total_gender = sum(self.gender_counts.values())
        if total_gender == 0:
            return "unknown"
        
        # Return gender with highest count
        return max(self.gender_counts, key=self.gender_counts.get)
    
    def get_dominant_age(self):
        """Get the most detected age group."""
        if not self.age_history:
            return "unknown"
        
        total_age = sum(self.age_counts.values())
        if total_age == 0:
            return "unknown"
        
        # Return age with highest count
        return max(self.age_counts, key=self.age_counts.get)
    
    def send_to_server(self, json_output, server_url="http://localhost:3000/pipeline"):
        """
        Send pipeline results to server via HTTP POST.
        
        Args:
            json_output: Dictionary with pipeline results
            server_url: Server endpoint URL
        """
        print("\n" + "=" * 70)
        print("SENDING DATA TO SERVER")
        print("=" * 70)
        print(f"Server URL: {server_url}")
        
        try:
            # Send POST request
            r = requests.post(server_url, data=json_output, timeout=10)
            
            print(f"Status Code: {r.status_code}")
            
            if r.status_code == 200:
                print("✓ Data sent successfully!")
                print(f"Server Response: {r.text}")
            else:
                print(f"⚠ Server returned status {r.status_code}")
                print(f"Response: {r.text}")
        
        except requests.exceptions.ConnectionError:
            print("✗ Connection Error: Could not connect to server")
            print(f"  Make sure the server is running at {server_url}")
        
        except requests.exceptions.Timeout:
            print("✗ Timeout Error: Server took too long to respond")
        
        except Exception as e:
            print(f"✗ Error sending data to server: {e}")
        
        print("=" * 70)
    
    def generate_json_output(self, processing_time, average_emotion, average_body_score):
        """
        Generate JSON output in the specified format.
        
        Returns:
            Dictionary with metrics and client metadata
        """
        # Calculate satisfaction rate
        satisfaction_rate = self.calculate_satisfaction_rate(self.emotions_in_roi, processing_time)
        
        # Get dominant gender
        dominant_gender = self.get_dominant_gender()
        
        # Get dominant age
        dominant_age = self.get_dominant_age()
        
        # Build JSON data
        data = {
            "id": 0,
            "counterid": self.counter_id,
            "metrics[satisfaction_rate]": f"{satisfaction_rate:.2f}",
            "metrics[processing_time]": str(int(processing_time)),
            "client_meta[age]": dominant_age if dominant_age != "unknown" else "19",
            "client_meta[gender]": dominant_gender
        }
        
        return data
    
    def run(self, display=True):
        """
        Run the integrated pipeline.
        
        Args:
            display: Whether to display video window
        
        Returns:
            Dictionary with JSON formatted results
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
            return None, None, None
        
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
            
            # Generate JSON output
            json_output = self.generate_json_output(processing_time, average_emotion, average_body_score)
            
            # Save JSON output to file
            output_file = "pipeline_output.json"
            try:
                with open(output_file, 'w') as f:
                    json.dump(json_output, f, indent=4)
                print(f"\n✓ Results saved to: {output_file}")
            except Exception as e:
                print(f"\n✗ Error saving JSON file: {e}")
            
            # Send data to server
            self.send_to_server(json_output)
            
            # Print final statistics
            self.print_final_statistics(elapsed_time, processing_time, average_emotion, average_body_score, json_output)
            
            return json_output
    
    def print_final_statistics(self, elapsed_time, processing_time, average_emotion, average_body_score, json_output):
        """Print comprehensive final statistics."""
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
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
        
        print()
        print("GENDER CLASSIFICATION:")
        print(f"  Total gender samples (in ROI): {len(self.gender_history)}")
        if self.gender_history:
            dominant_gender = self.get_dominant_gender()
            total_gender = sum(self.gender_counts.values())
            print(f"  Dominant gender: {dominant_gender.upper()}")
            print()
            print("  Gender distribution:")
            for gender, count in self.gender_counts.items():
                pct = (count / total_gender) * 100 if total_gender > 0 else 0
                bar = "█" * int(pct / 2)
                print(f"    {gender.capitalize():10s}: {count:4d} ({pct:5.1f}%) {bar}")
        else:
            print("  No gender data")
        
        print()
        print("AGE CLASSIFICATION:")
        print(f"  Total age samples (in ROI): {len(self.age_history)}")
        if self.age_history:
            dominant_age = self.get_dominant_age()
            total_age = sum(self.age_counts.values())
            print(f"  Dominant age group: {dominant_age}")
            print()
            print("  Age distribution:")
            sorted_ages = sorted(self.age_counts.items(), key=lambda x: x[1], reverse=True)
            for age, count in sorted_ages:
                if count > 0:
                    pct = (count / total_age) * 100 if total_age > 0 else 0
                    bar = "█" * int(pct / 2)
                    print(f"    {age:10s}: {count:4d} ({pct:5.1f}%) {bar}")
        else:
            print("  No age data")
        
        print("=" * 70)
        print()
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 20 + "JSON OUTPUT RESULTS" + " " * 29 + "║")
        print("╚" + "═" * 68 + "╝")
        print()
        print(json.dumps(json_output, indent=4))
        print()
        print("=" * 70)


def main():
    """Main entry point."""
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + " " * 18 + "INTEGRATED PIPELINE" + " " * 33 + "║")
    print("║" + " " * 2 + "YOLO + FER + Body Language + Gender + Age Classification" + " " * 11 + "║")
    print("╚" + "═" * 70 + "╝")
    print()
    
    if not TFLITE_AVAILABLE:
        print("⚠️  TensorFlow not available - body language analysis will be disabled")
        print("   Install with: pip install tensorflow")
        print()
    
    if not TRANSFORMERS_AVAILABLE:
        print("⚠️  Transformers not available - gender & age classification will be disabled")
        print("   Install with: pip install transformers torch")
        print()
    
    # Configuration
    yolo_model_path = r"../examples/yolov8n.pt"
    tflite_model_path = r"../models/body_language.tflite"
    
    # Choose video source
    # Option 1: Use webcam
    # video_path = None
    
    # Option 2: Use video file
    video_path = r"../data_manipulator/Data_sample_Time_processing_&_Emotion_Detection/sample_cam2.mp4"
    
    # Check if models exist
    if not os.path.exists(yolo_model_path):
        print(f"Error: YOLO model not found at {yolo_model_path}")
        return
    
    if not os.path.exists(tflite_model_path):
        print(f"Warning: TFLite model not found at {tflite_model_path}")
        print("Body language analysis will be disabled")
        tflite_model_path = None
    
    # Check if video exists (if using video file)
    if video_path and not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    
    # Initialize and run pipeline
    # Frame skipping optimized for smooth performance: YOLO(3), FER(8), Body(8), Gender(15), Age(15)
    pipeline = IntegratedPipeline(
        yolo_model_path, 
        tflite_model_path, 
        video_path, 
        yolo_skip_frames=3,      # More frequent person detection for smoother tracking
        fer_skip_frames=8,       # Balanced emotion detection
        body_skip_frames=8,      # Balanced body language
        gender_skip_frames=15,   # Less frequent (transformer models are slower)
        age_skip_frames=15,      # Less frequent (transformer models are slower)
        counter_id="C1"
    )
    
    json_result = pipeline.run(display=True)
    
    print("\n" + "=" * 70)
    print("✓ Pipeline execution completed successfully!")
    print("=" * 70)



if __name__ == "__main__":
    main()
