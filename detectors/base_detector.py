"""
Base detector interface.

This module defines the base class for all detector implementations.
Future tasks will implement concrete detectors (YOLO, MediaPipe, etc.)
that inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for all detectors.
    
    This class defines the interface that all detector implementations
    must follow to ensure consistency across different backends.
    """
    
    def __init__(self, config):
        """
        Initialize the detector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the detector model and resources.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Perform detection on the given frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            Dictionary containing detection results:
            {
                'faces': List of face detections with bounding boxes and landmarks,
                'poses': List of pose keypoints,
                'hands': List of hand landmarks (optional),
                'confidence': Overall detection confidence,
                'metadata': Additional detection metadata
            }
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the detector is available and ready to use.
        
        Returns:
            True if detector is available, False otherwise
        """
        pass
    
    def cleanup(self):
        """
        Clean up detector resources.
        
        This method should be called when the detector is no longer needed.
        """
        self.initialized = False
