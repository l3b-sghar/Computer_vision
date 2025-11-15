"""
Utilities package for common helper functions.

This package contains utility functions for video processing,
image preprocessing, validation, logging, and other common tasks.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Union


def load_video_source(source: Union[int, str]) -> cv2.VideoCapture:
    """
    Load video source (webcam or video file).
    
    Args:
        source: Video source (0 for webcam, or path to video file)
        
    Returns:
        OpenCV VideoCapture object
        
    Raises:
        ValueError: If video source cannot be opened
    """
    # Convert string to int if it's a digit
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {source}")
    
    return cap


def preprocess_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False
) -> np.ndarray:
    """
    Preprocess frame for detection/analysis.
    
    Args:
        frame: Input frame as numpy array
        target_size: Optional target size as (width, height)
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed frame
    """
    if frame is None or frame.size == 0:
        raise ValueError("Invalid frame")
    
    processed = frame.copy()
    
    # Resize if target size is specified
    if target_size is not None:
        processed = cv2.resize(processed, target_size)
    
    # Normalize if requested
    if normalize:
        processed = processed.astype(np.float32) / 255.0
    
    return processed


def draw_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box on frame.
    
    Args:
        frame: Input frame
        bbox: Bounding box as (x, y, width, height)
        label: Optional label text
        color: Box color as (B, G, R)
        thickness: Line thickness
        
    Returns:
        Frame with bounding box drawn
    """
    x, y, w, h = bbox
    annotated = frame.copy()
    
    # Draw rectangle
    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label if provided
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # Get text size
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw background for text
        cv2.rectangle(annotated, (x, y - text_h - 5), (x + text_w, y), color, -1)
        
        # Draw text
        cv2.putText(annotated, label, (x, y - 5), font, font_scale, (255, 255, 255), font_thickness)
    
    return annotated


def calculate_fps(timestamps: list, window_size: int = 30) -> float:
    """
    Calculate FPS from timestamps.
    
    Args:
        timestamps: List of timestamps
        window_size: Number of frames to consider for FPS calculation
        
    Returns:
        Current FPS
    """
    if len(timestamps) < 2:
        return 0.0
    
    recent_timestamps = timestamps[-window_size:]
    time_diff = recent_timestamps[-1] - recent_timestamps[0]
    
    if time_diff > 0:
        return (len(recent_timestamps) - 1) / time_diff
    
    return 0.0


__all__ = [
    'load_video_source',
    'preprocess_frame',
    'draw_bbox',
    'calculate_fps'
]
