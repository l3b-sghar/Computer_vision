"""
FER Library Demo - Video Analysis

This script analyzes facial emotion and sentiment from video files using the 'fer' library.
The 'fer' library is a lightweight alternative with built-in face detection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import numpy as np
import cv2
from fer.fer import FER


class FERLibraryVideoAnalyzer:
    """Video file emotion analysis using the fer library."""
    
    def __init__(self, video_path, output_path=None, show_preview=True):
        """
        Initialize video analyzer.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save annotated video
            show_preview: Whether to show live preview during processing
        """
        self.video_path = video_path
        self.output_path = output_path
        self.show_preview = show_preview
        self.detector = None
        self.cap = None
        self.writer = None
        
        # Statistics
        self.frame_count = 0
        self.total_frames = 0
        self.faces_detected = 0
        self.start_time = None
        
        # Emotion tracking
        self.emotion_history = []
        
        # Colors
        self.emotion_colors = {
            'happy': (0, 255, 0),
            'surprise': (0, 255, 255),
            'neutral': (255, 255, 255),
            'sad': (255, 0, 0),
            'angry': (0, 0, 255),
            'disgust': (128, 0, 128),
            'fear': (0, 165, 255)
        }
    
    def initialize(self) -> bool:
        """Initialize FER detector and video source."""
        print("Initializing FER Library Video Analyzer...")
        print("-" * 50)
        
        # Check video file
        if not os.path.exists(self.video_path):
            print(f"✗ Video file not found: {self.video_path}")
            return False
        print(f"✓ Video file found: {self.video_path}")
        
        # Initialize FER
        try:
            print("Loading FER detector...")
            self.detector = FER(mtcnn=False)
            print("✓ FER detector initialized")
        except Exception as e:
            print(f"✗ Failed to initialize FER: {e}")
            return False
        
        # Open video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"✗ Cannot open video: {self.video_path}")
            return False
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = self.total_frames / fps if fps > 0 else 0
        
        print(f"✓ Video opened: {width}x{height} @ {fps:.2f} FPS")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Duration: {duration:.2f} seconds")
        
        # Setup video writer
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            if self.writer.isOpened():
                print(f"✓ Output video: {self.output_path}")
            else:
                print(f"✗ Failed to create output writer")
                self.writer = None
        
        print("-" * 50)
        print()
        return True
    
    def draw_annotations(self, frame, faces_data):
        """Draw annotations on frame."""
        for face in faces_data:
            box = face['box']
            emotions = face['emotions']
            
            x, y, w, h = box
            
            # Get dominant emotion
            dominant = max(emotions.items(), key=lambda x: x[1])
            emotion = dominant[0]
            confidence = dominant[1]
            
            # Get color
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{emotion.upper()} ({confidence:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    def draw_progress(self, frame):
        """Draw progress bar."""
        height, width = frame.shape[:2]
        
        progress = self.frame_count / self.total_frames if self.total_frames > 0 else 0
        bar_width = int(width * 0.8)
        bar_x = int(width * 0.1)
        bar_y = height - 60
        bar_height = 20
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Progress
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                     (0, 255, 0), -1)
        
        # Text
        progress_text = f"{self.frame_count}/{self.total_frames} ({progress*100:.1f}%)"
        cv2.putText(frame, progress_text, (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Stats
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        eta = (self.total_frames - self.frame_count) / fps if fps > 0 else 0
        
        stats_text = f"Processing: {fps:.1f} FPS | ETA: {eta:.1f}s | Faces: {self.faces_detected}"
        cv2.putText(frame, stats_text, (bar_x, bar_y + bar_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def generate_report(self):
        """Generate comprehensive sentiment analysis report."""
        print("\n" + "=" * 70)
        print(" " * 20 + "SENTIMENT ANALYSIS REPORT")
        print("=" * 70)
        
        print(f"\nVideo: {os.path.basename(self.video_path)}")
        print(f"Frames processed: {self.frame_count}")
        print(f"Faces detected: {self.faces_detected}")
        
        if len(self.emotion_history) == 0:
            print("\nNo emotions detected in video")
            print("=" * 70)
            return
        
        # Count emotions
        emotion_counts = {}
        for emotions in self.emotion_history:
            dominant = max(emotions.items(), key=lambda x: x[1])[0]
            emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1
        
        # Overall analysis
        print(f"\n{'EMOTION DISTRIBUTION':^70}")
        print("-" * 70)
        total = len(self.emotion_history)
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            bar = "█" * int(percentage / 2)
            print(f"{emotion:10s}: {percentage:5.1f}% {bar}")
        
        # Key insights
        print(f"\n{'KEY INSIGHTS':^70}")
        print("-" * 70)
        
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        print(f"• Most expressed emotion: {dominant_emotion.upper()}")
        print(f"• Total emotion detections: {len(self.emotion_history)}")
        
        print("\n" + "=" * 70)
    
    def process_video(self):
        """Process the video file."""
        if not self.initialize():
            print("Failed to initialize analyzer")
            return
        
        print("Processing video...")
        print("Press 'Q' to stop early" if self.show_preview else "")
        print()
        
        self.start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("\nEnd of video reached")
                    break
                
                self.frame_count += 1
                
                # Detect emotions
                result = self.detector.detect_emotions(frame)
                
                # Process faces
                for face in result:
                    self.faces_detected += 1
                    self.emotion_history.append(face['emotions'])
                
                # Draw annotations
                self.draw_annotations(frame, result)
                self.draw_progress(frame)
                
                # Save to output
                if self.writer:
                    self.writer.write(frame)
                
                # Display preview
                if self.show_preview:
                    cv2.imshow('Video Analysis', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        print("\nStopping processing...")
                        break
                
                # Progress update
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / self.total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({self.frame_count}/{self.total_frames})")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            elapsed = time.time() - self.start_time
            print(f"\nProcessed in {elapsed:.2f} seconds")
            print(f"Average speed: {self.frame_count/elapsed:.2f} FPS")
            
            self.cap.release()
            if self.writer:
                self.writer.release()
                print(f"Annotated video saved: {self.output_path}")
            
            if self.show_preview:
                cv2.destroyAllWindows()
            
            self.generate_report()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Video FER and Sentiment Analysis using fer library',
        epilog="""
Examples:
  python fer_library_demo_video.py input.mp4
  python fer_library_demo_video.py input.mp4 --output result.mp4
  python fer_library_demo_video.py input.mp4 --no-preview
        """
    )
    
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to save annotated video')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable live preview')
    
    args = parser.parse_args()
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "FER LIBRARY - VIDEO SENTIMENT ANALYSIS" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("This uses the 'fer' library for emotion recognition")
    print("Install: pip install fer")
    print()
    
    analyzer = FERLibraryVideoAnalyzer(
        video_path=args.video,
        output_path=args.output,
        show_preview=not args.no_preview
    )
    analyzer.process_video()


if __name__ == "__main__":
    main()
