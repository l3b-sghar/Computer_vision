"""
Facial Emotion Recognition (FER) module.

This module implements facial emotion recognition using a lightweight
MobileNetV2-based model. It supports dual input modes (face crops or
FaceMesh landmarks) and provides temporal smoothing for stable predictions.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import cv2

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class FacialEmotionRecognizer:
    """
    Facial Emotion Recognition (FER) class.
    
    This class performs emotion recognition from face crops or FaceMesh landmarks.
    It uses a lightweight MobileNetV2-based model and includes temporal smoothing
    for stable predictions.
    
    Attributes:
        config: Configuration object
        model: TensorFlow/Keras model for emotion recognition
        emotion_labels: List of emotion labels
        emotion_history: History of emotion predictions for temporal smoothing
        smoothed_probabilities: Exponentially smoothed emotion probabilities
    """
    
    def __init__(self, config):
        """
        Initialize the FER module.
        
        Args:
            config: Configuration object containing FER parameters
        """
        self.config = config
        self.model = None
        self.emotion_labels = config.EMOTION_LABELS
        self.input_size = config.FER_INPUT_SIZE
        self.smoothing_alpha = config.FER_TEMPORAL_SMOOTHING_ALPHA
        self.confidence_threshold = config.FER_CONFIDENCE_THRESHOLD
        self.fallback_emotion = config.FER_FALLBACK_EMOTION
        
        # Temporal smoothing
        self.emotion_history: List[Dict[str, float]] = []
        self.smoothed_probabilities: Optional[np.ndarray] = None
        
        # Face detection cascade for preprocessing
        self.face_cascade = None
        
        if not TENSORFLOW_AVAILABLE:
            print("Warning: TensorFlow not available. FER will use fallback mode.")
    
    def initialize(self) -> bool:
        """
        Initialize the FER model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. FER initialization failed.")
            return False
        
        try:
            # Try to load a pre-trained model if available
            if self.config.FER_USE_PRETRAINED:
                try:
                    self.model = keras.models.load_model(self.config.EMOTION_MODEL_PATH)
                    print(f"Loaded pre-trained FER model from {self.config.EMOTION_MODEL_PATH}")
                    return True
                except (OSError, IOError):
                    print(f"Pre-trained model not found at {self.config.EMOTION_MODEL_PATH}")
                    print("Building lightweight FER model from scratch...")
            
            # Build a lightweight model using MobileNetV2
            self.model = self._build_fer_model()
            print("FER model initialized successfully (lightweight MobileNetV2-based)")
            
            return True
        except Exception as e:
            print(f"Error initializing FER model: {str(e)}")
            return False
    
    def _build_fer_model(self) -> keras.Model:
        """
        Build a lightweight FER model using MobileNetV2 backbone.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        input_layer = layers.Input(shape=(self.input_size, self.input_size, 3))
        
        # Use MobileNetV2 as backbone (lightweight and efficient)
        base_model = keras.applications.MobileNetV2(
            input_shape=(self.input_size, self.input_size, 3),
            include_top=False,
            weights=None,  # Don't load ImageNet weights for this lightweight implementation
            pooling='avg'
        )
        
        # Build the model
        x = base_model(input_layer, training=False)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(len(self.emotion_labels), activation='softmax')(x)
        
        model = keras.Model(inputs=input_layer, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def recognize_emotion(
        self,
        face_data: Union[np.ndarray, Dict[str, Any]],
        use_temporal_smoothing: bool = True
    ) -> Dict[str, Any]:
        """
        Recognize emotion from face data.
        
        Args:
            face_data: Either a face crop (numpy array) or a dictionary containing
                      face landmarks/mesh ('landmarks', 'face_crop', or 'bbox')
            use_temporal_smoothing: Whether to apply temporal smoothing
            
        Returns:
            Dictionary containing:
                - emotion: Detected emotion label
                - confidence: Confidence score (0-1)
                - probabilities: Dictionary of emotion probabilities
                - satisfaction_score_face: Satisfaction score based on emotion (0-1)
                - all_emotions: Dictionary with all 7 emotion probabilities
        """
        # Extract face crop from input
        face_crop = self._extract_face_crop(face_data)
        
        # Fallback if no face detected
        if face_crop is None:
            return self._get_fallback_result()
        
        # Preprocess face crop
        preprocessed = self._preprocess_face(face_crop)
        
        # Predict emotion
        if self.model is not None and TENSORFLOW_AVAILABLE:
            probabilities = self.model.predict(
                np.expand_dims(preprocessed, axis=0),
                verbose=0
            )[0]
        else:
            # Fallback: return neutral emotion with low confidence
            probabilities = np.zeros(len(self.emotion_labels))
            neutral_idx = self.emotion_labels.index('neutral')
            probabilities[neutral_idx] = 1.0
        
        # Apply temporal smoothing
        if use_temporal_smoothing:
            probabilities = self._apply_temporal_smoothing(probabilities)
        
        # Get dominant emotion
        emotion_idx = np.argmax(probabilities)
        emotion = self.emotion_labels[emotion_idx]
        confidence = float(probabilities[emotion_idx])
        
        # Create emotion probabilities dictionary
        emotion_probs = {
            label: float(prob)
            for label, prob in zip(self.emotion_labels, probabilities)
        }
        
        # Compute satisfaction score from emotion
        satisfaction_score = self._compute_satisfaction_score(emotion_probs)
        
        # Store in history
        result = {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': emotion_probs,
            'satisfaction_score_face': satisfaction_score,
            'all_emotions': emotion_probs
        }
        self.emotion_history.append(result)
        
        return result
    
    def _extract_face_crop(
        self,
        face_data: Union[np.ndarray, Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """
        Extract face crop from various input formats.
        
        Args:
            face_data: Face crop array or dictionary with face information
            
        Returns:
            Face crop as numpy array, or None if extraction fails
        """
        # Case 1: Direct face crop (numpy array)
        if isinstance(face_data, np.ndarray):
            return face_data
        
        # Case 2: Dictionary with face_crop key
        if isinstance(face_data, dict):
            if 'face_crop' in face_data and face_data['face_crop'] is not None:
                return face_data['face_crop']
            
            # Case 3: Dictionary with bbox - extract from frame
            if 'bbox' in face_data and 'frame' in face_data:
                bbox = face_data['bbox']
                frame = face_data['frame']
                x, y, w, h = bbox
                return frame[y:y+h, x:x+w]
            
            # Case 4: Dictionary with FaceMesh landmarks
            if 'landmarks' in face_data or 'face_landmarks' in face_data:
                landmarks = face_data.get('landmarks')
                if landmarks is None:
                    landmarks = face_data.get('face_landmarks')
                frame = face_data.get('frame')
                
                if landmarks is not None and frame is not None:
                    # Extract bounding box from landmarks
                    bbox = self._landmarks_to_bbox(landmarks, frame.shape)
                    if bbox is not None:
                        x, y, w, h = bbox
                        return frame[y:y+h, x:x+w]
        
        return None
    
    def _landmarks_to_bbox(
        self,
        landmarks: Union[np.ndarray, List],
        frame_shape: Tuple[int, int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Convert facial landmarks to bounding box.
        
        Args:
            landmarks: Facial landmarks as array or list
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            Bounding box as (x, y, w, h) or None
        """
        try:
            if isinstance(landmarks, list):
                landmarks = np.array(landmarks)
            
            if len(landmarks) == 0:
                return None
            
            # Extract x and y coordinates
            if landmarks.ndim == 2 and landmarks.shape[1] >= 2:
                x_coords = landmarks[:, 0]
                y_coords = landmarks[:, 1]
            else:
                return None
            
            # Compute bounding box with padding
            min_x = int(np.min(x_coords))
            max_x = int(np.max(x_coords))
            min_y = int(np.min(y_coords))
            max_y = int(np.max(y_coords))
            
            # Add padding (10%)
            padding_x = int((max_x - min_x) * 0.1)
            padding_y = int((max_y - min_y) * 0.1)
            
            x = max(0, min_x - padding_x)
            y = max(0, min_y - padding_y)
            w = min(frame_shape[1] - x, max_x - min_x + 2 * padding_x)
            h = min(frame_shape[0] - y, max_y - min_y + 2 * padding_y)
            
            return (x, y, w, h)
        except Exception:
            return None
    
    def _preprocess_face(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess face crop for model input.
        
        Args:
            face_crop: Face crop as numpy array
            
        Returns:
            Preprocessed face crop
        """
        # Convert to grayscale if needed, then back to RGB for MobileNet
        if len(face_crop.shape) == 2:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
        elif face_crop.shape[2] == 4:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGRA2RGB)
        elif face_crop.shape[2] == 3:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(face_crop, (self.input_size, self.input_size))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def _apply_temporal_smoothing(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply exponential moving average for temporal smoothing.
        
        Args:
            probabilities: Current emotion probabilities
            
        Returns:
            Smoothed probabilities
        """
        if self.smoothed_probabilities is None:
            self.smoothed_probabilities = probabilities
        else:
            self.smoothed_probabilities = (
                self.smoothing_alpha * probabilities +
                (1 - self.smoothing_alpha) * self.smoothed_probabilities
            )
        
        return self.smoothed_probabilities
    
    def _compute_satisfaction_score(self, emotion_probs: Dict[str, float]) -> float:
        """
        Compute satisfaction score from emotion probabilities.
        
        Maps emotions to satisfaction values:
        - happy: 1.0 (very satisfied)
        - surprise: 0.7 (moderately positive)
        - neutral: 0.5 (neither satisfied nor dissatisfied)
        - fear: 0.4 (slightly negative)
        - sad: 0.3 (dissatisfied)
        - disgust: 0.2 (very dissatisfied)
        - anger: 0.1 (extremely dissatisfied)
        
        Args:
            emotion_probs: Dictionary of emotion probabilities
            
        Returns:
            Satisfaction score (0-1)
        """
        emotion_to_satisfaction = {
            'happy': 1.0,
            'surprise': 0.7,
            'neutral': 0.5,
            'fear': 0.4,
            'sad': 0.3,
            'disgust': 0.2,
            'anger': 0.1
        }
        
        # Compute weighted average
        satisfaction_score = 0.0
        for emotion, prob in emotion_probs.items():
            if emotion in emotion_to_satisfaction:
                satisfaction_score += prob * emotion_to_satisfaction[emotion]
        
        return float(satisfaction_score)
    
    def _get_fallback_result(self) -> Dict[str, Any]:
        """
        Get fallback result when no face is detected.
        
        Returns:
            Dictionary with fallback emotion and scores
        """
        probabilities = {label: 0.0 for label in self.emotion_labels}
        probabilities[self.fallback_emotion] = 1.0
        
        return {
            'emotion': self.fallback_emotion,
            'confidence': 0.0,
            'probabilities': probabilities,
            'satisfaction_score_face': 0.5,  # Neutral satisfaction
            'all_emotions': probabilities
        }
    
    def get_emotion_history(self, window_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get emotion history.
        
        Args:
            window_size: Number of recent frames to return (None for all)
            
        Returns:
            List of emotion recognition results
        """
        if window_size is None:
            return self.emotion_history
        return self.emotion_history[-window_size:]
    
    def compute_average_emotion(self, window_size: int = 10) -> Dict[str, Any]:
        """
        Compute average emotion over a time window.
        
        Args:
            window_size: Number of recent frames to consider
            
        Returns:
            Dictionary with average emotion statistics
        """
        if len(self.emotion_history) == 0:
            return {
                'dominant_emotion': self.fallback_emotion,
                'average_confidence': 0.0,
                'average_satisfaction': 0.5,
                'emotion_distribution': {label: 0.0 for label in self.emotion_labels}
            }
        
        # Get recent history
        recent = self.emotion_history[-window_size:]
        
        # Compute average probabilities
        avg_probs = {label: 0.0 for label in self.emotion_labels}
        avg_confidence = 0.0
        avg_satisfaction = 0.0
        
        for result in recent:
            for label, prob in result['probabilities'].items():
                avg_probs[label] += prob
            avg_confidence += result['confidence']
            avg_satisfaction += result['satisfaction_score_face']
        
        # Normalize
        n = len(recent)
        avg_probs = {label: prob / n for label, prob in avg_probs.items()}
        avg_confidence /= n
        avg_satisfaction /= n
        
        # Get dominant emotion
        dominant_emotion = max(avg_probs, key=avg_probs.get)
        
        return {
            'dominant_emotion': dominant_emotion,
            'average_confidence': avg_confidence,
            'average_satisfaction': avg_satisfaction,
            'emotion_distribution': avg_probs
        }
    
    def reset(self):
        """Reset the FER state including history and smoothing."""
        self.emotion_history = []
        self.smoothed_probabilities = None
    
    def is_available(self) -> bool:
        """
        Check if FER is available and ready.
        
        Returns:
            True if FER is initialized and ready, False otherwise
        """
        return TENSORFLOW_AVAILABLE and self.model is not None
