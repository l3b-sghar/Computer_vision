"""
Detectors package for emotion and body-language detection.

This package contains detector implementations for face detection,
pose estimation, and emotion recognition using various backends
(YOLO, MediaPipe, etc.).
"""

from typing import Protocol, Tuple, Optional, Dict, Any
import numpy as np


class DetectorProtocol(Protocol):
    """
    Protocol defining the interface for all detectors.
    
    This ensures consistency across different detector implementations
    (YOLO, MediaPipe, custom detectors, etc.).
    """
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect features in the given frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            Dictionary containing detection results with keys:
            - 'faces': List of face bounding boxes and landmarks
            - 'poses': List of pose keypoints
            - 'confidence': Detection confidence scores
        """
        ...
    
    def is_available(self) -> bool:
        """
        Check if the detector is available and properly initialized.
        
        Returns:
            True if detector is ready to use, False otherwise
        """
        ...


__all__ = ['DetectorProtocol']
