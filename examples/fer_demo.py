"""
FER Module Demo

This script demonstrates how to use the Facial Emotion Recognition (FER) module
with various input formats and integration scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from config import Config
from analytics import FacialEmotionRecognizer


def demo_basic_usage():
    """Demonstrate basic FER usage."""
    print("=" * 70)
    print("DEMO 1: Basic FER Usage")
    print("=" * 70)
    
    # Initialize FER
    fer = FacialEmotionRecognizer(Config)
    success = fer.initialize()
    
    if not success:
        print("Failed to initialize FER")
        return
    
    # Create a sample face crop
    face_crop = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    
    # Recognize emotion
    result = fer.recognize_emotion(face_crop)
    
    # Display results
    print(f"Detected Emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Satisfaction Score: {result['satisfaction_score_face']:.3f}")
    print("\nAll Emotion Probabilities:")
    for emotion, prob in result['probabilities'].items():
        print(f"  {emotion:10s}: {prob:.3f}")
    print()


def demo_multiple_input_formats():
    """Demonstrate FER with different input formats."""
    print("=" * 70)
    print("DEMO 2: Multiple Input Formats")
    print("=" * 70)
    
    fer = FacialEmotionRecognizer(Config)
    fer.initialize()
    
    # Format 1: Direct face crop
    print("Format 1: Direct face crop (NumPy array)")
    face_crop = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    result = fer.recognize_emotion(face_crop)
    print(f"  Result: {result['emotion']} (confidence: {result['confidence']:.3f})")
    print()
    
    # Format 2: Dictionary with face_crop
    print("Format 2: Dictionary with face_crop")
    face_data = {'face_crop': face_crop}
    result = fer.recognize_emotion(face_data)
    print(f"  Result: {result['emotion']} (confidence: {result['confidence']:.3f})")
    print()
    
    # Format 3: Dictionary with bbox
    print("Format 3: Dictionary with bounding box")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    face_data = {
        'bbox': (100, 100, 100, 100),
        'frame': frame
    }
    result = fer.recognize_emotion(face_data)
    print(f"  Result: {result['emotion']} (confidence: {result['confidence']:.3f})")
    print()
    
    # Format 4: Dictionary with landmarks
    print("Format 4: Dictionary with FaceMesh landmarks")
    landmarks = np.random.randint(100, 200, (68, 2))
    face_data = {
        'landmarks': landmarks,
        'frame': frame
    }
    result = fer.recognize_emotion(face_data)
    print(f"  Result: {result['emotion']} (confidence: {result['confidence']:.3f})")
    print()


def demo_temporal_smoothing():
    """Demonstrate temporal smoothing."""
    print("=" * 70)
    print("DEMO 3: Temporal Smoothing")
    print("=" * 70)
    
    fer = FacialEmotionRecognizer(Config)
    fer.initialize()
    fer.reset()
    
    print("Processing 10 frames with temporal smoothing...")
    print()
    
    for i in range(10):
        face_crop = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        result = fer.recognize_emotion(face_crop, use_temporal_smoothing=True)
        
        print(f"Frame {i+1:2d}: {result['emotion']:10s} "
              f"(confidence: {result['confidence']:.3f}, "
              f"satisfaction: {result['satisfaction_score_face']:.3f})")
    print()
    
    # Show average emotion
    avg_result = fer.compute_average_emotion(window_size=10)
    print(f"Average over 10 frames:")
    print(f"  Dominant emotion: {avg_result['dominant_emotion']}")
    print(f"  Average confidence: {avg_result['average_confidence']:.3f}")
    print(f"  Average satisfaction: {avg_result['average_satisfaction']:.3f}")
    print()


def demo_detector_integration():
    """Demonstrate integration with detector output."""
    print("=" * 70)
    print("DEMO 4: Integration with Detector Output")
    print("=" * 70)
    
    fer = FacialEmotionRecognizer(Config)
    fer.initialize()
    
    # Simulate detector output (like from MediaPipe or YOLO)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detector_output = {
        'faces': [
            {
                'bbox': (150, 100, 120, 140),
                'confidence': 0.98,
                'landmarks': np.random.randint(150, 270, (68, 2))
            },
            {
                'bbox': (400, 150, 100, 120),
                'confidence': 0.95,
                'landmarks': np.random.randint(400, 500, (68, 2))
            }
        ],
        'frame': frame
    }
    
    print(f"Detected {len(detector_output['faces'])} faces")
    print()
    
    # Process each detected face
    for idx, face in enumerate(detector_output['faces'], 1):
        face_data = {
            'bbox': face['bbox'],
            'landmarks': face['landmarks'],
            'frame': detector_output['frame']
        }
        
        result = fer.recognize_emotion(face_data)
        
        print(f"Face {idx}:")
        print(f"  Detection confidence: {face['confidence']:.3f}")
        print(f"  Emotion: {result['emotion']}")
        print(f"  Emotion confidence: {result['confidence']:.3f}")
        print(f"  Satisfaction score: {result['satisfaction_score_face']:.3f}")
        print()


def demo_fallback_behavior():
    """Demonstrate fallback behavior."""
    print("=" * 70)
    print("DEMO 5: Fallback Behavior (No Face Detected)")
    print("=" * 70)
    
    fer = FacialEmotionRecognizer(Config)
    fer.initialize()
    
    # Empty input (no face)
    result = fer.recognize_emotion({})
    
    print("No face detected in frame")
    print(f"Fallback emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Satisfaction score: {result['satisfaction_score_face']:.3f}")
    print()


def demo_satisfaction_mapping():
    """Demonstrate emotion to satisfaction score mapping."""
    print("=" * 70)
    print("DEMO 6: Emotion to Satisfaction Score Mapping")
    print("=" * 70)
    
    emotion_to_satisfaction = {
        'happy': 1.0,
        'surprise': 0.7,
        'neutral': 0.5,
        'fear': 0.4,
        'sad': 0.3,
        'disgust': 0.2,
        'anger': 0.1
    }
    
    print("Emotion       -> Satisfaction Score")
    print("-" * 40)
    for emotion, score in emotion_to_satisfaction.items():
        bar = "█" * int(score * 30)
        print(f"{emotion:10s} -> {score:.1f} {bar}")
    print()


def main():
    """Run all demos."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "FER MODULE DEMONSTRATION" + " " * 29 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    try:
        demo_basic_usage()
        demo_multiple_input_formats()
        demo_temporal_smoothing()
        demo_detector_integration()
        demo_fallback_behavior()
        demo_satisfaction_mapping()
        
        print("=" * 70)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print()
        
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
