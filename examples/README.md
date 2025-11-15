# Examples

This directory contains example scripts demonstrating facial emotion recognition (FER) and sentiment analysis.

## Available Demos

### 1. Live Camera Demos

#### `fer_demo_live.py` - Custom FER Implementation (DeepFace)
Real-time emotion recognition using our custom FER module with DeepFace backend.

#### `fer_library_demo_live.py` - FER Library
Real-time emotion recognition using the lightweight `fer` library.

### 2. Video Analysis Demos

#### `fer_demo_video.py` - Custom FER Implementation (DeepFace)
Analyzes pre-recorded videos and generates comprehensive sentiment reports.

#### `fer_library_demo_video.py` - FER Library
Video analysis using the lightweight `fer` library.

---

## 1. Custom FER Implementation (DeepFace Backend)

### Quick Setup (RECOMMENDED)

Install DeepFace for accurate emotion recognition:
```powershell
pip install deepface numpy opencv-python
```

### Running the Demo

```powershell
cd examples
python fer_demo.py
```

### What it does

- **Real-time webcam emotion recognition** with 7 emotions: happy, sad, angry, surprise, fear, disgust, neutral
- **Live visualization** with colored bounding boxes, emotion labels, and probability bars
- **Temporal smoothing** for stable predictions
- **Interactive controls**: Q (quit), R (reset), S (statistics)

### Troubleshooting "Bad" Emotion Detection

If emotions are always neutral or seem random:

1. **Install DeepFace** (most important):
   ```powershell
   pip install deepface
   ```

2. **Check lighting**: Ensure your face is well-lit

3. **Face the camera**: Look directly at the camera

4. **Wait for initialization**: First run downloads models (~50-100MB)

### Requirements

**Minimal (Testing Only - Returns Neutral)**:
- numpy
- opencv-python

**Recommended (Accurate Emotions)**:
- deepface ← **Install this for real emotion detection!**
- numpy
- opencv-python

### Performance

- **With DeepFace**: 15-30 FPS with accurate emotion recognition
- **Without DeepFace**: Fast but only returns "neutral" (fallback mode)

### Alternative Pre-trained Models

If you prefer different options:

**HSEmotion** (Fast & Accurate):
```powershell
pip install hsemotion
```

**FER Package** (Simple):
```powershell
pip install fer
```

### What the Demo Shows

- Real-time face detection using OpenCV Haar Cascade
- Emotion recognition with confidence scores
- Satisfaction scores (0-1) based on emotions
- Emotion probability distributions
- FPS counter and session statistics
- Emotion distribution over time

### Live Camera Usage

```powershell
python fer_demo_live.py
```

### Video Analysis Usage

```powershell
# Analyze video without saving output
python fer_demo_video.py input.mp4

# Analyze and save annotated video
python fer_demo_video.py input.mp4 --output result.mp4

# Process without preview (faster)
python fer_demo_video.py input.mp4 --no-preview
```

### Output Example

When working correctly with DeepFace, you'll see:
- Colored boxes around detected faces
- Emotion labels (e.g., "HAPPY (0.89)")
- Satisfaction score (e.g., "Satisfaction: 0.92")
- Probability bars for all 7 emotions
- Real-time FPS and face count
- Comprehensive sentiment report (video analysis)

---

## 2. FER Library Implementation

Lightweight alternative using the `fer` library with built-in face detection.

### Quick Setup

```powershell
pip install fer
```

### Live Camera Usage

```powershell
python fer_library_demo_live.py
```

### Video Analysis Usage

```powershell
# Analyze video
python fer_library_demo_video.py input.mp4

# With output video
python fer_library_demo_video.py input.mp4 --output result.mp4

# No preview (faster)
python fer_library_demo_video.py input.mp4 --no-preview
```

### Features

- **Simpler setup** - just one pip install
- **Built-in face detection** - uses Haar Cascade or MTCNN
- **Good performance** - 20-40 FPS on webcam
- **7 emotions detected** - same as DeepFace version
- **Satisfaction scoring** - emotion to sentiment mapping
- **Comprehensive reports** - detailed video analysis

---

## Comparison: Which Demo to Use?

| Feature | Custom FER (DeepFace) | FER Library |
|---------|----------------------|-------------|
| **Accuracy** | ★★★★★ Very High | ★★★★☆ High |
| **Speed** | ★★★★☆ Good (15-30 FPS) | ★★★★★ Fast (20-40 FPS) |
| **Setup** | pip install deepface | pip install fer |
| **Dependencies** | Heavy (TensorFlow) | Light |
| **Best For** | Production, High Accuracy | Quick tests, Demos |

---

## All Demo Files

```
examples/
├── fer_demo_live.py              # Custom FER + DeepFace (Live)
├── fer_demo_video.py             # Custom FER + DeepFace (Video)
├── fer_library_demo_live.py      # FER Library (Live)
├── fer_library_demo_video.py     # FER Library (Video)
├── fer_demo_old.py               # Old demo (backup)
└── README.md                     # This file
```
