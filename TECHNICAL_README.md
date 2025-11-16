# Computer Vision Customer Analytics System

A comprehensive computer vision system for analyzing customer engagement, emotions, and behavior at service counters using multiple detection models and tracking algorithms.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Technology Choices](#technology-choices)
  - [YOLO for Person Detection](#yolo-for-person-detection)
  - [FER Library for Emotion Analysis](#fer-library-for-emotion-analysis)
  - [TFLite Model for Body Language](#tflite-model-for-body-language)
- [Components](#components)
  - [Examples](#examples)
  - [Full Pipeline](#full-pipeline)
  - [Fine-Tuned Version](#fine-tuned-version)
  - [Data Manipulation](#data-manipulation)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Optimization](#performance-optimization)
- [License](#license)

---

## Overview

This system provides multi-modal customer behavior analysis combining:
- **Person Detection & Tracking** (YOLO)
- **Facial Emotion Recognition** (FER Library)
- **Body Language Analysis** (TFLite Custom Model)
- **Region of Interest (ROI) Tracking** (Custom Implementation)
- **Time-based Analytics** (Session tracking)

The system is designed for retail/service environments to analyze customer engagement, satisfaction, and processing times at service counters.

---

## Project Structure

```
Computer_vision/
├── examples/                          # Individual demo scripts
│   ├── fer_library_demo_live.py      # Real-time emotion recognition
│   ├── yolo_test_demo.py             # YOLO object detection baseline
│   ├── yolo_roi_tracker.py           # YOLO with ROI time tracking
│   ├── body_language_demo.py         # Face-based engagement analysis
│   └── body_language_tflite_demo.py  # Custom TFLite body language model
├── full_pipeline/                     # Integrated system
│   └── main.py                        # YOLO + FER integrated pipeline
├── fine_tuned_version/               # Custom fine-tuned YOLO
│   ├── best.pt                        # Fine-tuned weights
│   ├── best.onnx                      # ONNX export
│   └── test.py                        # Fine-tuned inference script
├── data_manipulator/                 # Dataset preparation tools
│   └── extract_frames_from_video.py  # Frame extraction for annotation
├── models/                           # Model weights
│   ├── yolo11s.pt                    # YOLOv11 baseline
│   └── body_language.tflite          # Custom body language model
├── analytics/                        # Analysis modules (legacy)
├── detectors/                        # Detector classes (legacy)
└── utils/                            # Utilities (legacy)
```

---

## Technology Choices

### YOLO for Person Detection

**Why YOLO?**
- ⚡ **Real-time Performance**: 7-15 FPS on CPU, 30+ FPS on GPU
- 🎯 **High Accuracy**: State-of-the-art object detection for person class
- 🔧 **Easy Fine-tuning**: Can train on custom classes (personFF, personFB, counter)
- 📦 **Pre-trained Weights**: Strong baseline with COCO dataset (80 classes)
- 🔄 **Active Development**: YOLOv11 is the latest version with improvements

**Use Cases in This Project:**
1. **Baseline Detection** (`yolo_test_demo.py`): Detects persons using standard COCO classes
2. **ROI Tracking** (`yolo_roi_tracker.py`): Detects persons and tracks time in region of interest
3. **Fine-tuned Model** (`fine_tuned_version/test.py`): Custom classes for orientation detection
   - `personFF`: Person Facing Forward (engaging with counter)
   - `personFB`: Person Facing Backward (leaving/not engaged)
   - `counter`: Counter/service area detection

**Alternatives Considered:**
- MediaPipe Holistic: Not available in Python 3.13
- OpenCV Haar Cascades: Less accurate, outdated
- Faster R-CNN: Slower inference time

---

### FER Library for Emotion Analysis

**Why FER Library?**
- 🚀 **Lightweight**: Minimal dependencies, fast inference
- 📊 **7 Emotion Classes**: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- 🎭 **Good Baseline**: Reasonable accuracy without complex setup
- 🔌 **Simple API**: Easy integration with OpenCV pipelines
- 💻 **CPU-Friendly**: Works well without GPU acceleration

**Technical Details:**
- Uses OpenCV Haar Cascade for face detection (mtcnn=False for speed)
- Can optionally use MTCNN for better face detection (mtcnn=True, slower)
- Returns emotion probabilities as dictionary
- Supports real-time video processing

**Implementation:**
```python
from fer.fer import FER
detector = FER(mtcnn=False)
emotions = detector.detect_emotions(frame)
# Returns: [{'box': (x,y,w,h), 'emotions': {'happy': 0.8, ...}}]
```

**Use Cases in This Project:**
1. **Live Demo** (`fer_library_demo_live.py`): Real-time webcam emotion detection
2. **Video Analysis** (`fer_library_demo_video.py`): Process recorded videos
3. **Integrated Pipeline** (`full_pipeline/main.py`): Combined with YOLO for ROI-based emotion tracking

**Alternatives Considered:**
- DeepFace: Heavier, more dependencies, slower
- Custom CNN: Requires large dataset and training infrastructure
- Azure/AWS APIs: Cloud dependency, cost, latency

---

### TFLite Model for Body Language

**Why Custom TFLite Model?**
- 🎯 **Specialized Training**: Trained on specific body language emotions
- ⚡ **Fast Inference**: TensorFlow Lite optimized for edge devices
- 🧠 **Rich Features**: Uses MediaPipe pose landmarks (2004 features)
- 🎨 **9 Emotion Classes**: Happy, Sad, Angry, Surprised, Confused, Tension, Excited, Pain, Depressed

**Model Architecture:**
- **Input**: [1, 2004] - MediaPipe pose landmark coordinates
  - 33 body landmarks × 4 coordinates (x, y, z, visibility)
  - Additional derived features
- **Output**: [1, 9] - Probability distribution over 9 emotions
- **Framework**: TensorFlow Lite (converted from trained TensorFlow model)

**Why Body Language Matters:**
- 💬 **Non-verbal Communication**: 55% of communication is non-verbal
- 🔍 **Contextual Understanding**: Face emotions + body posture = better accuracy
- 🚨 **Early Detection**: Body language often precedes facial expression changes
- 📈 **Engagement Metrics**: Posture reveals interest, stress, comfort levels

**Current Implementation:**
- `body_language_tflite_demo.py`: Demo using pixel sampling (workaround)
- Real implementation requires MediaPipe Pose (Python 3.8-3.12)
- 72 FPS performance with current workaround

**Ideal Setup:**
```python
import mediapipe as mp
mp_pose = mp.solutions.pose.Pose()
landmarks = mp_pose.process(frame).pose_landmarks
# Extract 2004 features from landmarks
prediction = tflite_model.predict(features)
```

**Alternatives Considered:**
- OpenPose: Heavyweight, complex setup
- AlphaPose: Requires GPU, slower
- Rule-based System: Less accurate, hard to maintain

---

## Components

### Examples

Individual demo scripts for testing and development:

1. **`fer_library_demo_live.py`**
   - Real-time facial emotion recognition from webcam
   - 7 emotions: happy, sad, angry, surprise, fear, disgust, neutral
   - Statistics tracking and visualization
   - ~33 FPS performance

2. **`yolo_test_demo.py`**
   - YOLO baseline person detection
   - 80 COCO classes including person
   - Confidence threshold: 0.5
   - ~7 FPS on CPU

3. **`yolo_roi_tracker.py`**
   - Person detection with ROI (Region of Interest)
   - ROI: Bottom 1/5th of frame (counter area)
   - Automatic time tracking when person in ROI
   - IoU calculation for overlap detection
   - Session-based analytics

4. **`body_language_demo.py`**
   - Face-based engagement analysis (fallback mode)
   - Posture classification: leaning forward/back, standing, sitting
   - ~49 FPS performance
   - Works without MediaPipe (Python 3.13 compatible)

5. **`body_language_tflite_demo.py`**
   - Custom TFLite model for body language/emotion
   - 9 emotion classes from body posture
   - ~72 FPS with pixel sampling workaround
   - Needs MediaPipe for full accuracy

### Full Pipeline

**`full_pipeline/main.py`** - Integrated YOLO + FER System

**Purpose**: Complete customer analytics at service counter

**Features**:
- YOLO person detection with ROI tracking
- FER emotion analysis only for persons in ROI
- Crops person bounding box for focused emotion detection
- Time tracking: total time, session time, entries
- Frame skipping optimization (configurable)
- Real-time FPS monitoring

**Performance Optimizations**:
- YOLO runs every 2nd frame (configurable: `yolo_skip_frames`)
- FER runs every 3rd frame (configurable: `fer_skip_frames`)
- Results cached between skipped frames
- Expected performance: 15-25 FPS (3-5x speedup)

**Output**:
```python
processing_time, average_emotion = pipeline.run()
# processing_time: float (seconds in ROI)
# average_emotion: dict (average emotion probabilities)
```

**Configuration**:
```python
pipeline = IntegratedPipeline(
    yolo_model_path="../models/yolo11s.pt",
    video_path="video.mp4",  # or None for webcam
    yolo_skip_frames=2,       # Process every 2nd frame
    fer_skip_frames=3         # Process every 3rd frame
)
```

### Fine-Tuned Version

**`fine_tuned_version/test.py`** - Custom YOLO with Orientation Detection

**Custom Classes**:
- `personFF`: Person Facing Forward (engaged with counter)
- `personFB`: Person Facing Backward (leaving/disengaged)
- `counter`: Counter/service area

**Key Innovation**:
- Timer starts ONLY when personFF overlaps with counter
- Distinguishes between approaching and leaving customers
- More accurate engagement tracking than simple ROI

**Training Data**:
- Custom dataset annotated with Roboflow
- ~1000 images from service counter videos
- Fine-tuned from YOLOv11s baseline

**Use Case**:
Perfect for retail/service environments where customer orientation matters:
- Queue management
- Service time analytics
- Customer engagement scoring
- Staff performance metrics

### Data Manipulation

**`extract_frames_from_video.py`** - Dataset Preparation Tool

**Features**:
- Extracts frames from video for annotation
- Configurable frame sampling (default: every 20th frame)
- Incremental extraction (doesn't overwrite existing frames)
- Progress tracking

**Usage**:
```python
video_path = "sample_video.mp4"
output_folder = "data/"
save_interval = 20  # Save 1 every 20 frames
```

**Output**:
- Saved frames: `frame_00000.jpg`, `frame_00020.jpg`, etc.
- Shows: processed count vs saved count
- Continues numbering from existing frames

---

## Installation

### Prerequisites
- Python 3.8+ (3.13 for some features, 3.8-3.12 for MediaPipe)
- pip package manager

### Install Dependencies

```bash
# Core dependencies
pip install opencv-python numpy ultralytics

# Emotion recognition
pip install fer

# TensorFlow Lite (for body language model)
pip install tensorflow

# Optional: MediaPipe (Python 3.8-3.12 only)
pip install mediapipe
```

### Quick Start

```bash
# Clone repository
git clone https://github.com/l3b-sghar/Computer_vision.git
cd Computer_vision

# Install requirements
pip install -r requirements.txt

# Run demos
python examples/fer_library_demo_live.py
python examples/yolo_roi_tracker.py
python full_pipeline/main.py
```

---

## Usage

### Running Individual Demos

```bash
# Facial Emotion Recognition
cd examples
python fer_library_demo_live.py

# YOLO Person Detection
python yolo_test_demo.py

# YOLO with ROI Tracking
python yolo_roi_tracker.py

# Body Language Analysis
python body_language_tflite_demo.py
```

### Running Full Pipeline

```bash
cd full_pipeline
python main.py
```

**For webcam**: Set `video_path = None` in the script  
**For video file**: Set `video_path = "path/to/video.mp4"`

### Running Fine-Tuned Model

```bash
cd fine_tuned_version
python test.py
```

Requires `best.pt` (fine-tuned weights) in the same directory.

### Frame Extraction for Dataset

```bash
cd data_manipulator
python extract_frames_from_video.py
```

Modify script to set:
- `video_path`: Source video
- `output_folder`: Destination folder
- `save_interval`: Frame sampling rate

---

## Performance Optimization

### Frame Skipping Strategy

The integrated pipeline uses intelligent frame skipping:

```python
# Process YOLO every 2nd frame
if frame_count % yolo_skip_frames == 0:
    yolo_results = model.detect(frame)
else:
    yolo_results = last_cached_result

# Process FER every 3rd frame (only when person in ROI)
if person_in_roi and frame_count % fer_skip_frames == 0:
    emotions = fer.detect(cropped_person)
```

**Benefits**:
- 3-5x speed improvement
- Minimal accuracy loss (person movement is relatively slow)
- Configurable trade-off between speed and accuracy

### Hardware Recommendations

**Minimum**:
- CPU: Intel i5 or equivalent
- RAM: 8GB
- Performance: 5-10 FPS

**Recommended**:
- CPU: Intel i7 or AMD Ryzen 7
- GPU: NVIDIA GTX 1660 or better
- RAM: 16GB
- Performance: 20-30 FPS

**Optimal**:
- CPU: Intel i9 or AMD Ryzen 9
- GPU: NVIDIA RTX 3060 or better
- RAM: 32GB
- Performance: 50+ FPS

### GPU Acceleration

For YOLO GPU acceleration:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Model Comparison

| Model | Purpose | Speed (CPU) | Accuracy | Use Case |
|-------|---------|-------------|----------|----------|
| YOLOv11s | Person Detection | ~7 FPS | High | Baseline detection |
| FER Library | Emotion Recognition | ~30 FPS | Medium-High | Real-time emotions |
| Body Language TFLite | Posture Analysis | ~72 FPS | High* | Body language |
| Fine-tuned YOLO | Orientation Detection | ~7 FPS | Very High | Custom engagement |

*Requires MediaPipe pose landmarks for full accuracy

---

## Future Improvements

### Short-term
- [ ] Integrate MediaPipe Pose for body language TFLite
- [ ] Add attention tracking (gaze direction)
- [ ] Implement multi-person tracking
- [ ] Add emotion timeline visualization
- [ ] Export analytics to CSV/JSON

### Long-term
- [ ] Train custom emotion classifier for "interested/confused"
- [ ] Implement staff-customer interaction detection
- [ ] Add queue management features
- [ ] Deploy as REST API service
- [ ] Mobile app integration
- [ ] Real-time dashboard for analytics

---

## Citation

If you use this system in your research or project, please cite:

```bibtex
@software{computer_vision_analytics,
  author = {l3b-sghar},
  title = {Computer Vision Customer Analytics System},
  year = {2025},
  url = {https://github.com/l3b-sghar/Computer_vision}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Contact

**Repository Owner**: l3b-sghar  
**Repository**: [Computer_vision](https://github.com/l3b-sghar/Computer_vision)

For questions, issues, or feature requests, please open an issue in the repository.

---

## Acknowledgments

- **YOLO**: Ultralytics YOLOv11
- **FER Library**: fer package for emotion recognition
- **MediaPipe**: Google's pose detection framework
- **TensorFlow**: Model training and inference
- **OpenCV**: Computer vision operations

---

**Last Updated**: November 16, 2025
