"""
Configuration module for the emotion and body-language analyzer.

This module contains all configuration settings for the system,
including detection thresholds, model paths, and output settings.
"""

import os
from typing import Dict, Any


class Config:
    """
    Central configuration class for the emotion analyzer system.
    
    Attributes:
        DETECTOR_TYPE: Type of detector to use ('auto', 'yolo', 'mediapipe')
        VIDEO_SOURCE: Video source (0 for webcam, or path to video file)
        FRAME_WIDTH: Target frame width for processing
        FRAME_HEIGHT: Target frame height for processing
        FPS: Frames per second for processing
        CONFIDENCE_THRESHOLD: Minimum confidence threshold for detections
    """
    
    # Detector Configuration
    DETECTOR_TYPE = os.getenv('DETECTOR_TYPE', 'auto')  # 'auto', 'yolo', 'mediapipe'
    
    # Video Input Configuration
    VIDEO_SOURCE = os.getenv('VIDEO_SOURCE', '0')  # 0 for webcam or path to video file
    FRAME_WIDTH = int(os.getenv('FRAME_WIDTH', '640'))
    FRAME_HEIGHT = int(os.getenv('FRAME_HEIGHT', '480'))
    FPS = int(os.getenv('FPS', '30'))
    
    # Detection Configuration
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
    FACE_DETECTION_CONFIDENCE = float(os.getenv('FACE_DETECTION_CONFIDENCE', '0.5'))
    POSE_DETECTION_CONFIDENCE = float(os.getenv('POSE_DETECTION_CONFIDENCE', '0.5'))
    
    # YOLO Configuration
    YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'models/yolo/yolov8n.pt')
    YOLO_CONFIDENCE = float(os.getenv('YOLO_CONFIDENCE', '0.5'))
    
    # MediaPipe Configuration
    MEDIAPIPE_MODEL_COMPLEXITY = int(os.getenv('MEDIAPIPE_MODEL_COMPLEXITY', '1'))
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE = float(os.getenv('MEDIAPIPE_MIN_DETECTION_CONFIDENCE', '0.5'))
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE = float(os.getenv('MEDIAPIPE_MIN_TRACKING_CONFIDENCE', '0.5'))
    
    # Emotion Recognition Configuration
    EMOTION_MODEL_PATH = os.getenv('EMOTION_MODEL_PATH', 'models/emotion/emotion_model.h5')
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # FER Configuration
    FER_INPUT_SIZE = int(os.getenv('FER_INPUT_SIZE', '48'))  # Input size for FER model
    FER_USE_PRETRAINED = os.getenv('FER_USE_PRETRAINED', 'true').lower() == 'true'
    FER_TEMPORAL_SMOOTHING_ALPHA = float(os.getenv('FER_TEMPORAL_SMOOTHING_ALPHA', '0.3'))  # EMA smoothing factor
    FER_CONFIDENCE_THRESHOLD = float(os.getenv('FER_CONFIDENCE_THRESHOLD', '0.5'))  # Minimum confidence for predictions
    FER_FALLBACK_EMOTION = os.getenv('FER_FALLBACK_EMOTION', 'neutral')  # Default emotion when no face detected
    
    # Analytics Configuration
    SATISFACTION_WEIGHT_FACE = float(os.getenv('SATISFACTION_WEIGHT_FACE', '0.6'))
    SATISFACTION_WEIGHT_BODY = float(os.getenv('SATISFACTION_WEIGHT_BODY', '0.4'))
    SMOOTHING_WINDOW_SIZE = int(os.getenv('SMOOTHING_WINDOW_SIZE', '10'))
    
    # Temporal Tracking
    INTERACTION_START_THRESHOLD = float(os.getenv('INTERACTION_START_THRESHOLD', '2.0'))  # seconds
    INTERACTION_END_THRESHOLD = float(os.getenv('INTERACTION_END_THRESHOLD', '3.0'))  # seconds
    
    # Output Configuration
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')
    SAVE_ANNOTATED_VIDEO = os.getenv('SAVE_ANNOTATED_VIDEO', 'false').lower() == 'true'
    DISPLAY_VISUALIZATION = os.getenv('DISPLAY_VISUALIZATION', 'true').lower() == 'true'
    
    # Optional Scores Configuration
    ENABLE_ATTENTION_SCORE = os.getenv('ENABLE_ATTENTION_SCORE', 'true').lower() == 'true'
    ENABLE_STRESS_SCORE = os.getenv('ENABLE_STRESS_SCORE', 'true').lower() == 'true'
    ENABLE_HESITANCY_SCORE = os.getenv('ENABLE_HESITANCY_SCORE', 'true').lower() == 'true'
    ENABLE_BODY_LANGUAGE_EVENTS = os.getenv('ENABLE_BODY_LANGUAGE_EVENTS', 'true').lower() == 'true'
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.
        
        Returns:
            Dictionary containing all configuration values
        """
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values to update
        """
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Check required paths exist if specified
        if cls.DETECTOR_TYPE not in ['auto', 'yolo', 'mediapipe']:
            print(f"Warning: Invalid DETECTOR_TYPE '{cls.DETECTOR_TYPE}', using 'auto'")
            cls.DETECTOR_TYPE = 'auto'
        
        # Check thresholds are in valid range
        if not 0.0 <= cls.CONFIDENCE_THRESHOLD <= 1.0:
            print(f"Warning: CONFIDENCE_THRESHOLD {cls.CONFIDENCE_THRESHOLD} out of range, using 0.5")
            cls.CONFIDENCE_THRESHOLD = 0.5
        
        # Create output directory if it doesn't exist
        if not os.path.exists(cls.OUTPUT_DIR):
            os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        
        return True


# Validate configuration on module import
Config.validate()
