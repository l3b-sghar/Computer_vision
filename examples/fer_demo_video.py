"""
Video File FER Demo

This script analyzes facial emotion and sentiment from pre-recorded video files.
It processes the video frame-by-frame, detects faces, recognizes emotions,
and generates comprehensive sentiment analysis reports.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
from pathlib import Path
import numpy as np
import cv2
from config import Config
from analytics import FacialEmotionRecognizer


class VideoFERAnalyzer:
    """Video file facial emotion recognition and sentiment analysis."""
    
    def __init__(self, video_path, output_path=None, show_preview=True):
        """
        Initialize video FER analyzer.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save annotated video
            show_preview: Whether to show live preview during processing
        """
        self.video_path = video_path
        self.output_path = output_path
        self.show_preview = show_preview
        self.fer = None
        self.face_cascade = None
        self.cap = None
        self.writer = None
        
        # Statistics
        self.frame_count = 0
        self.total_frames = 0
        self.faces_detected = 0
        self.start_time = None
        
        # Sentiment tracking
        self.emotion_timeline = []
        self.satisfaction_timeline = []
        self.frame_timestamps = []
        
        # Colors for visualization
        self.emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'surprise': (0, 255, 255),  # Yellow
            'neutral': (255, 255, 255), # White
            'sad': (255, 0, 0),         # Blue
            'angry': (0, 0, 255),       # Red
            'disgust': (128, 0, 128),   # Purple
            'fear': (0, 165, 255)       # Orange
        }
    
    def initialize(self) -> bool:
        """
        Initialize FER module and video source.
        
        Returns:
            True if initialization successful
        """
        print("Initializing Video FER Analyzer...")
        print("-" * 50)
        
        # Check if video file exists
        if not os.path.exists(self.video_path):
            print(f"✗ Video file not found: {self.video_path}")
            return False
        print(f"✓ Video file found: {self.video_path}")
        
        # Initialize FER
        self.fer = FacialEmotionRecognizer(Config)
        if not self.fer.initialize():
            print("Warning: FER initialized in fallback mode")
        else:
            print("✓ FER module initialized")
        
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("✗ Failed to load face detector")
            return False
        print("✓ Face detector loaded")
        
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
        
        # Setup video writer if output path specified
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            if self.writer.isOpened():
                print(f"✓ Output video will be saved to: {self.output_path}")
            else:
                print(f"✗ Failed to create output video writer")
                self.writer = None
        
        print("-" * 50)
        print()
        return True
    
    def detect_faces(self, frame):
        """
        Detect faces in frame using Haar Cascade.
        
        Args:
            frame: Input frame
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        return faces
    
    def draw_annotations(self, frame, faces_data):
        """
        Draw annotations on frame.
        
        Args:
            frame: Frame to draw on
            faces_data: List of (bbox, emotion_result) tuples
        """
        for (x, y, w, h), emotion_result in faces_data:
            emotion = emotion_result['emotion']
            confidence = emotion_result['confidence']
            satisfaction = emotion_result['satisfaction_score_face']
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw emotion label
            label = f"{emotion.upper()} ({confidence:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Background for label
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 10, y), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw satisfaction score
            satisfaction_text = f"Satisfaction: {satisfaction:.2f}"
            cv2.putText(frame, satisfaction_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_progress(self, frame):
        """
        Draw progress bar and statistics.
        
        Args:
            frame: Frame to draw on
        """
        height, width = frame.shape[:2]
        
        # Progress bar
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
        
        # Progress text
        progress_text = f"{self.frame_count}/{self.total_frames} ({progress*100:.1f}%)"
        cv2.putText(frame, progress_text, (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Processing stats
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        eta = (self.total_frames - self.frame_count) / fps if fps > 0 else 0
        
        stats_text = f"Processing: {fps:.1f} FPS | ETA: {eta:.1f}s | Faces: {self.faces_detected}"
        cv2.putText(frame, stats_text, (bar_x, bar_y + bar_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def generate_report(self):
        """Generate and display comprehensive sentiment analysis report."""
        print("\n" + "=" * 70)
        print(" " * 20 + "SENTIMENT ANALYSIS REPORT")
        print("=" * 70)
        
        # Video information
        print(f"\nVideo: {os.path.basename(self.video_path)}")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Faces detected: {self.faces_detected}")
        
        if len(self.fer.emotion_history) == 0:
            print("\nNo emotions detected in video")
            print("=" * 70)
            return
        
        # Overall emotion analysis
        avg_result = self.fer.compute_average_emotion(window_size=len(self.fer.emotion_history))
        
        print(f"\n{'OVERALL EMOTION ANALYSIS':^70}")
        print("-" * 70)
        print(f"Dominant emotion: {avg_result['dominant_emotion'].upper()}")
        print(f"Average confidence: {avg_result['average_confidence']:.3f}")
        print(f"Average satisfaction: {avg_result['average_satisfaction']:.3f}")
        
        # Emotion distribution
        print(f"\n{'EMOTION DISTRIBUTION':^70}")
        print("-" * 70)
        for emotion, prob in sorted(avg_result['emotion_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 40)
            percentage = prob * 100
            print(f"{emotion:10s}: {percentage:5.1f}% {bar}")
        
        # Sentiment classification
        print(f"\n{'OVERALL SENTIMENT':^70}")
        print("-" * 70)
        satisfaction = avg_result['average_satisfaction']
        
        if satisfaction >= 0.7:
            sentiment = "POSITIVE ✓"
            color_code = "😊"
        elif satisfaction >= 0.4:
            sentiment = "NEUTRAL ●"
            color_code = "😐"
        else:
            sentiment = "NEGATIVE ✗"
            color_code = "😞"
        
        print(f"Sentiment: {sentiment} {color_code}")
        print(f"Satisfaction Score: {satisfaction:.3f} / 1.000")
        
        # Temporal analysis
        if len(self.satisfaction_timeline) > 0:
            print(f"\n{'TEMPORAL ANALYSIS':^70}")
            print("-" * 70)
            
            # Split video into segments
            segment_size = len(self.satisfaction_timeline) // 5
            if segment_size > 0:
                segments = ['Beginning', 'Early', 'Middle', 'Late', 'End']
                for i, seg_name in enumerate(segments):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size if i < 4 else len(self.satisfaction_timeline)
                    
                    if start_idx < len(self.satisfaction_timeline):
                        seg_satisfaction = np.mean(self.satisfaction_timeline[start_idx:end_idx])
                        bar = "█" * int(seg_satisfaction * 30)
                        print(f"{seg_name:10s}: {seg_satisfaction:.3f} {bar}")
        
        # Key insights
        print(f"\n{'KEY INSIGHTS':^70}")
        print("-" * 70)
        
        # Most common emotion
        print(f"• Most expressed emotion: {avg_result['dominant_emotion'].upper()}")
        
        # Emotional variability
        if len(self.satisfaction_timeline) > 1:
            variability = np.std(self.satisfaction_timeline)
            if variability < 0.1:
                print(f"• Emotional stability: VERY STABLE (σ={variability:.3f})")
            elif variability < 0.2:
                print(f"• Emotional stability: STABLE (σ={variability:.3f})")
            else:
                print(f"• Emotional stability: VARIABLE (σ={variability:.3f})")
        
        # Peak satisfaction
        if len(self.satisfaction_timeline) > 0:
            max_satisfaction = max(self.satisfaction_timeline)
            min_satisfaction = min(self.satisfaction_timeline)
            print(f"• Peak satisfaction: {max_satisfaction:.3f}")
            print(f"• Lowest satisfaction: {min_satisfaction:.3f}")
            print(f"• Satisfaction range: {max_satisfaction - min_satisfaction:.3f}")
        
        print("\n" + "=" * 70)
    
    def process_video(self):
        """Process the video file and analyze emotions."""
        if not self.initialize():
            print("Failed to initialize analyzer")
            return
        
        print("Processing video...")
        print("Press 'Q' to stop processing early" if self.show_preview else "")
        print()
        
        self.start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("\nEnd of video reached")
                    break
                
                self.frame_count += 1
                frame_time = self.frame_count / self.cap.get(cv2.CAP_PROP_FPS)
                self.frame_timestamps.append(frame_time)
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process each face
                faces_data = []
                for (x, y, w, h) in faces:
                    self.faces_detected += 1
                    
                    # Extract face region
                    face_crop = frame[y:y+h, x:x+w]
                    
                    # Recognize emotion
                    emotion_result = self.fer.recognize_emotion(
                        face_crop, 
                        use_temporal_smoothing=True
                    )
                    
                    faces_data.append(((x, y, w, h), emotion_result))
                    
                    # Track for analysis
                    self.emotion_timeline.append({
                        'frame': self.frame_count,
                        'time': frame_time,
                        'emotion': emotion_result['emotion'],
                        'confidence': emotion_result['confidence']
                    })
                    self.satisfaction_timeline.append(emotion_result['satisfaction_score_face'])
                
                # Draw annotations
                self.draw_annotations(frame, faces_data)
                self.draw_progress(frame)
                
                # Save to output video if specified
                if self.writer:
                    self.writer.write(frame)
                
                # Display preview if enabled
                if self.show_preview:
                    cv2.imshow('Video FER Analysis', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        print("\nStopping processing...")
                        break
                
                # Progress update every 100 frames
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / self.total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({self.frame_count}/{self.total_frames} frames)")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            elapsed = time.time() - self.start_time
            print(f"\nProcessing completed in {elapsed:.2f} seconds")
            print(f"Average processing speed: {self.frame_count/elapsed:.2f} FPS")
            
            self.cap.release()
            if self.writer:
                self.writer.release()
                print(f"Annotated video saved to: {self.output_path}")
            
            if self.show_preview:
                cv2.destroyAllWindows()
            
            # Generate report
            self.generate_report()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Video FER and Sentiment Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fer_demo_video.py input.mp4
  python fer_demo_video.py input.mp4 --output annotated_output.mp4
  python fer_demo_video.py input.mp4 --no-preview
        """
    )
    
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to save annotated video (optional)')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable live preview (faster processing)')
    
    args = parser.parse_args()
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "VIDEO FER & SENTIMENT ANALYSIS" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Create and run analyzer
    analyzer = VideoFERAnalyzer(
        video_path=args.video,
        output_path=args.output,
        show_preview=not args.no_preview
    )
    analyzer.process_video()


if __name__ == "__main__":
    main()
