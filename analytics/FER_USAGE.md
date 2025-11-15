# Facial Emotion Recognition (FER) Module Usage Guide

## Overview

The FER module (`analytics/fer.py`) provides facial emotion recognition capabilities using a lightweight MobileNetV2-based model. It supports multiple input formats and includes temporal smoothing for stable predictions.

## Features

- **7 Emotion Classes**: happy, neutral, sad, anger, surprise, fear, disgust
- **Dual Input Mode**: Accepts face crops OR FaceMesh landmarks
- **Temporal Smoothing**: Exponential moving average for stable predictions
- **Fallback Behavior**: Handles cases when no face is detected
- **Satisfaction Score**: Computes satisfaction_score_face (0-1) from emotions
- **Integration Ready**: Compatible with auto_selector and detector outputs

## Quick Start

```python
from config import Config
from analytics import FacialEmotionRecognizer

# Initialize FER
fer = FacialEmotionRecognizer(Config)
fer.initialize()

# Recognize emotion from face crop
import numpy as np
face_crop = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
result = fer.recognize_emotion(face_crop)

print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Satisfaction: {result['satisfaction_score_face']:.3f}")
```

## Input Formats

### 1. Direct Face Crop (NumPy Array)

```python
# RGB or BGR image array
face_crop = cv2.imread('face.jpg')
result = fer.recognize_emotion(face_crop)
```

### 2. Dictionary with Face Crop

```python
face_data = {
    'face_crop': face_crop,
    'confidence': 0.95
}
result = fer.recognize_emotion(face_data)
```

### 3. Dictionary with Bounding Box

```python
face_data = {
    'bbox': (x, y, w, h),  # Bounding box coordinates
    'frame': frame         # Full frame image
}
result = fer.recognize_emotion(face_data)
```

### 4. Dictionary with FaceMesh Landmarks

```python
face_data = {
    'landmarks': landmarks,  # NumPy array of shape (N, 2)
    'frame': frame           # Full frame image
}
result = fer.recognize_emotion(face_data)
```

## Output Format

The `recognize_emotion()` method returns a dictionary with:

```python
{
    'emotion': 'happy',                    # Detected emotion label
    'confidence': 0.85,                    # Confidence score (0-1)
    'satisfaction_score_face': 0.92,       # Satisfaction score (0-1)
    'probabilities': {                     # All emotion probabilities
        'happy': 0.85,
        'neutral': 0.08,
        'sad': 0.03,
        'anger': 0.02,
        'surprise': 0.01,
        'fear': 0.01,
        'disgust': 0.00
    },
    'all_emotions': { ... }                # Same as probabilities
}
```

## Emotion to Satisfaction Mapping

The satisfaction score is computed using the following mapping:

- **happy**: 1.0 (very satisfied)
- **surprise**: 0.7 (moderately positive)
- **neutral**: 0.5 (neither satisfied nor dissatisfied)
- **fear**: 0.4 (slightly negative)
- **sad**: 0.3 (dissatisfied)
- **disgust**: 0.2 (very dissatisfied)
- **anger**: 0.1 (extremely dissatisfied)

## Temporal Smoothing

Enable temporal smoothing for stable predictions over time:

```python
# Process multiple frames with smoothing
for frame in video_frames:
    face_crop = extract_face(frame)
    result = fer.recognize_emotion(face_crop, use_temporal_smoothing=True)
```

Smoothing uses exponential moving average (EMA) with configurable alpha:

```python
# In config.py
FER_TEMPORAL_SMOOTHING_ALPHA = 0.3  # Lower = more smoothing
```

## Temporal Analysis

Get average emotion over a time window:

```python
# Compute average emotion over last 10 frames
avg_result = fer.compute_average_emotion(window_size=10)

print(f"Dominant emotion: {avg_result['dominant_emotion']}")
print(f"Average confidence: {avg_result['average_confidence']:.3f}")
print(f"Average satisfaction: {avg_result['average_satisfaction']:.3f}")
```

## Fallback Behavior

When no face is detected, FER returns a fallback result:

```python
# No face detected
result = fer.recognize_emotion({})

# Returns:
# {
#     'emotion': 'neutral',
#     'confidence': 0.0,
#     'satisfaction_score_face': 0.5,
#     ...
# }
```

Configure fallback emotion:

```python
# In config.py
FER_FALLBACK_EMOTION = 'neutral'  # Default emotion when no face detected
```

## Integration with Detectors

### With MediaPipe Detector

```python
from detectors.mediapipe_detector import MediaPipeDetector
from analytics import FacialEmotionRecognizer

detector = MediaPipeDetector(Config)
fer = FacialEmotionRecognizer(Config)

detector.initialize()
fer.initialize()

# Process frame
frame = cv2.imread('image.jpg')
detection_result = detector.detect(frame)

# Extract face data
if detection_result['faces']:
    face = detection_result['faces'][0]
    face_data = {
        'landmarks': face.get('landmarks'),
        'bbox': face.get('bbox'),
        'frame': frame
    }
    
    # Recognize emotion
    emotion_result = fer.recognize_emotion(face_data)
```

### With Auto Selector Output

The FER module is designed to work seamlessly with auto_selector outputs:

```python
# Auto selector determines best detection method
detection_output = auto_selector.detect(frame)

# FER processes the output
for face in detection_output['faces']:
    face_data = {
        'face_crop': face.get('face_crop'),
        'landmarks': face.get('landmarks'),
        'bbox': face.get('bbox'),
        'frame': frame
    }
    
    emotion_result = fer.recognize_emotion(face_data)
```

## Configuration Parameters

Add to `config.py`:

```python
# FER Configuration
FER_INPUT_SIZE = 48                      # Input size for FER model
FER_USE_PRETRAINED = True                # Use pre-trained model if available
FER_TEMPORAL_SMOOTHING_ALPHA = 0.3       # EMA smoothing factor (0-1)
FER_CONFIDENCE_THRESHOLD = 0.5           # Minimum confidence for predictions
FER_FALLBACK_EMOTION = 'neutral'         # Default emotion when no face detected
```

## Advanced Usage

### Reset State

```python
# Reset emotion history and smoothing
fer.reset()
```

### Get Emotion History

```python
# Get all emotion history
history = fer.get_emotion_history()

# Get recent history (last 5 frames)
recent = fer.get_emotion_history(window_size=5)
```

### Check Availability

```python
if fer.is_available():
    # FER is ready to use
    result = fer.recognize_emotion(face_crop)
else:
    # FER not available (e.g., TensorFlow not installed)
    print("FER not available")
```

## Performance Considerations

- **Input Size**: Default 48x48 pixels provides good balance between speed and accuracy
- **Model**: Lightweight MobileNetV2 backbone for fast inference
- **Temporal Smoothing**: Reduces noise but adds slight latency (configurable)
- **Batch Processing**: Process multiple faces in parallel for better throughput

## Notes

- The model is untrained by default. For production use, train on FER2013, RAF-DB, or AffectNet datasets
- Pre-trained models can be loaded by placing them at the configured `EMOTION_MODEL_PATH`
- The module gracefully handles missing TensorFlow installation
- All 7 emotions are from standard FER datasets (no "interested" emotion)

## References

- FER2013 Dataset: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
- RAF-DB Dataset: http://www.whdeng.cn/raf/model1.html
- MobileNetV2: https://arxiv.org/abs/1801.04381
