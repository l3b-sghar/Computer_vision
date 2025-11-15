"""
Main entry point for the emotion and body-language analyzer.

This module provides the main execution flow for analyzing customer
satisfaction and body language from video input (webcam or file).
"""

import sys
import time
import argparse
from typing import Optional, Dict, Any
import cv2

from config import Config
from utils import load_video_source, preprocess_frame, calculate_fps
from analytics import AnalyticsResult


class EmotionAnalyzer:
    """
    Main analyzer class for emotion and body-language detection.
    
    This class coordinates the detection pipeline, analytics computation,
    and result generation. It supports both YOLO and MediaPipe backends
    depending on configuration and video characteristics.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the emotion analyzer.
        
        Args:
            config: Optional configuration object (uses default Config if None)
        """
        self.config = config if config is not None else Config
        self.detector = None
        self.analyzer = None
        self.interaction_start_time = None
        self.frame_timestamps = []
        
        print("Initializing Emotion Analyzer...")
        print(f"Detector type: {self.config.DETECTOR_TYPE}")
        print(f"Video source: {self.config.VIDEO_SOURCE}")
        
    def _initialize_detector(self):
        """
        Initialize the appropriate detector based on configuration.
        
        This method will be implemented in future tasks to support
        YOLO, MediaPipe, or automatic detection.
        """
        # Placeholder for detector initialization
        # Will be implemented in future tasks
        print("Detector initialization will be implemented in future tasks")
        pass
    
    def _initialize_analyzer(self):
        """
        Initialize the analytics module.
        
        This method will be implemented in future tasks to support
        emotion analysis, body language interpretation, and score computation.
        """
        # Placeholder for analyzer initialization
        # Will be implemented in future tasks
        print("Analyzer initialization will be implemented in future tasks")
        pass
    
    def process_frame(self, frame) -> Dict[str, Any]:
        """
        Process a single frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary containing detection and analysis results
        """
        # Record timestamp
        current_time = time.time()
        self.frame_timestamps.append(current_time)
        
        # Track interaction start
        if self.interaction_start_time is None:
            self.interaction_start_time = current_time
        
        # Preprocess frame
        processed_frame = preprocess_frame(
            frame,
            target_size=(self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT)
        )
        
        # Placeholder for actual detection and analysis
        # Will be implemented in future tasks
        result = {
            'frame': processed_frame,
            'timestamp': current_time,
            'fps': calculate_fps(self.frame_timestamps),
            'detections': None,
            'emotions': None,
            'body_language': None
        }
        
        return result
    
    def compute_analytics(self) -> AnalyticsResult:
        """
        Compute final analytics results.
        
        Returns:
            AnalyticsResult object containing all computed scores
        """
        # Calculate processing time
        if self.interaction_start_time is not None:
            processing_time = time.time() - self.interaction_start_time
        else:
            processing_time = 0.0
        
        # Placeholder for actual analytics computation
        # Will be implemented in future tasks
        result = AnalyticsResult(
            customer_satisfaction_score=0.0,  # Placeholder
            processing_time_seconds=processing_time,
            attention_score=0.0 if self.config.ENABLE_ATTENTION_SCORE else None,
            stress_score=0.0 if self.config.ENABLE_STRESS_SCORE else None,
            hesitancy_score=0.0 if self.config.ENABLE_HESITANCY_SCORE else None,
            body_language_events=[] if self.config.ENABLE_BODY_LANGUAGE_EVENTS else None
        )
        
        return result
    
    def run(self):
        """
        Main execution loop.
        
        Processes video frames, performs detection and analysis,
        and generates final results.
        """
        print("\nStarting analysis...")
        print("Press 'q' to quit\n")
        
        try:
            # Load video source
            cap = load_video_source(self.config.VIDEO_SOURCE)
            
            # Initialize detector and analyzer
            self._initialize_detector()
            self._initialize_analyzer()
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("End of video stream or cannot read frame")
                    break
                
                # Process frame
                result = self.process_frame(frame)
                frame_count += 1
                
                # Display visualization if enabled
                if self.config.DISPLAY_VISUALIZATION:
                    display_frame = result['frame']
                    
                    # Add FPS counter
                    fps_text = f"FPS: {result['fps']:.1f}"
                    cv2.putText(display_frame, fps_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('Emotion Analyzer', display_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopping analysis...")
                    break
            
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Compute final analytics
            analytics = self.compute_analytics()
            
            # Display results
            print("\n" + "="*50)
            print("ANALYSIS RESULTS")
            print("="*50)
            print(f"Frames processed: {frame_count}")
            print(f"Processing time: {analytics.processing_time_seconds:.2f} seconds")
            print(f"Customer satisfaction score: {analytics.customer_satisfaction_score:.3f}")
            
            if analytics.attention_score is not None:
                print(f"Attention score: {analytics.attention_score:.3f}")
            
            if analytics.stress_score is not None:
                print(f"Stress score: {analytics.stress_score:.3f}")
            
            if analytics.hesitancy_score is not None:
                print(f"Hesitancy score: {analytics.hesitancy_score:.3f}")
            
            if analytics.body_language_events:
                print(f"Body language events: {len(analytics.body_language_events)}")
            
            print("="*50)
            
            return analytics
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return None
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Emotion and Body-Language Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Use default webcam
  python main.py --source 0         # Use webcam 0
  python main.py --source video.mp4 # Use video file
  python main.py --detector yolo    # Force YOLO detector
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default=None,
        help='Video source (0 for webcam, or path to video file)'
    )
    
    parser.add_argument(
        '--detector',
        type=str,
        choices=['auto', 'yolo', 'mediapipe'],
        default=None,
        help='Detector type to use'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable visualization display'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=None,
        help='Target frame width'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=None,
        help='Target frame height'
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point.
    """
    print("="*50)
    print("EMOTION AND BODY-LANGUAGE ANALYZER")
    print("="*50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Update configuration from arguments
    if args.source is not None:
        Config.VIDEO_SOURCE = args.source
    
    if args.detector is not None:
        Config.DETECTOR_TYPE = args.detector
    
    if args.no_display:
        Config.DISPLAY_VISUALIZATION = False
    
    if args.width is not None:
        Config.FRAME_WIDTH = args.width
    
    if args.height is not None:
        Config.FRAME_HEIGHT = args.height
    
    # Validate configuration
    if not Config.validate():
        print("Configuration validation failed")
        sys.exit(1)
    
    # Create and run analyzer
    analyzer = EmotionAnalyzer(Config)
    result = analyzer.run()
    
    if result is not None:
        print("\nAnalysis completed successfully")
        sys.exit(0)
    else:
        print("\nAnalysis failed or was interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
