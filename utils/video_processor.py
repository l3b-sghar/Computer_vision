"""
Video processing utilities.

This module provides utilities for video input/output handling,
frame processing, and video format conversions.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Generator
import time


class VideoProcessor:
    """
    Video processor for handling video input and output.
    
    This class provides methods for reading from video sources,
    writing to video files, and processing video streams.
    """
    
    def __init__(self, source: str, output_path: Optional[str] = None):
        """
        Initialize video processor.
        
        Args:
            source: Video source (camera index or file path)
            output_path: Optional output video path
        """
        self.source = source
        self.output_path = output_path
        self.cap = None
        self.writer = None
        self.fps = 30
        self.frame_width = 640
        self.frame_height = 480
    
    def open(self) -> bool:
        """
        Open video source.
        
        Returns:
            True if successful, False otherwise
        """
        # Convert string to int if it's a digit
        source = self.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            return False
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize writer if output path is specified
        if self.output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height)
            )
        
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the video source.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
        
        return self.cap.read()
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the output video.
        
        Args:
            frame: Frame to write
            
        Returns:
            True if successful, False otherwise
        """
        if self.writer is None:
            return False
        
        self.writer.write(frame)
        return True
    
    def frame_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames from the video source.
        
        Yields:
            Video frames as numpy arrays
        """
        while True:
            ret, frame = self.read_frame()
            if not ret:
                break
            yield frame
    
    def close(self):
        """Release video resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
