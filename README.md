# Computer Vision Customer Analytics System

A comprehensive multi-modal customer analytics system that combines person detection, facial emotion recognition, body language analysis, demographic classification, and temporal tracking for retail and service environments.

---

## Overview

This system provides real-time customer behavior analysis at service counters using a single RGB camera. It combines multiple state-of-the-art models to deliver comprehensive analytics:

- **Satisfaction Score** (0-1): Multi-modal fusion of facial emotions and body language
- **Processing Time**: Accurate session tracking in Region of Interest (ROI)
- **Demographics**: Age group and gender classification
- **Engagement Metrics**: Emotion timeline, body posture, and attention tracking

The system is optimized for real-time performance with intelligent frame skipping and model caching.

---

## Core Features

### 🎯 Person Detection & ROI Tracking
- **YOLO v8/v11**: State-of-the-art real-time person detection
- **Region of Interest (ROI)**: Automatic detection of counter interaction zone
- **Session Tracking**: Precise timing when customers enter/exit ROI
- **Adaptive ROI**: Configurable ROI size (20% default, 40% for specific cameras)

### 😊 Facial Emotion Recognition
- **FER Library**: 7-emotion classification (happy, sad, angry, surprise, fear, disgust, neutral)
- **MTCNN Face Detection**: High-accuracy face localization
- **Temporal Smoothing**: Emotion timeline analysis for robust satisfaction scoring
- **ROI-Focused**: Only analyzes emotions when person is in service area

### 🧍 Body Language Analysis
- **TFLite Model**: Custom-trained 9-class body posture classifier
- **Classes**: Happy, Sad, Angry, Surprised, Confused, Tension, Excited, Pain, Depressed
- **Satisfaction Mapping**: Converts posture to 0-100 satisfaction score
- **Lightweight**: Optimized for real-time edge inference

### 👤 Demographic Classification
- **Gender Classification**: 
  - Model: `rizvandwiki/gender-classification` (Hugging Face)
  - Accuracy: ~92% on UTKFace dataset
  - Classes: Male, Female
  
- **Age Classification**:
  - Model: `nateraw/vit-age-classifier` (Vision Transformer)
  - Accuracy: ~77% on UTKFace dataset
  - Classes: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+

### ⏱️ Processing Time Tracking
- **Entry/Exit Detection**: Automatic session start/stop
- **Multiple Sessions**: Tracks total time across all ROI entries
- **Millisecond Precision**: Accurate timing for analytics
- **Session Metadata**: Duration, entry count, average time per session

### 🔄 Multi-Modal Fusion
- **Satisfaction Rate**: Weighted combination of facial emotions (positive/negative)
- **Body Language Integration**: Posture scores complement emotion analysis
- **Temporal Context**: Earlier emotions weighted differently than recent ones
- **Confidence Scoring**: Model predictions include confidence levels

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Video Input (Camera/File)                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              YOLO Person Detection (YOLOv8n)                 │
│           • Detects persons in frame                         │
│           • Calculates ROI overlap (IOU)                     │
│           • Frame skipping: Every 3rd frame                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                    Person in ROI?
                         │
            ┌────────────┴────────────┐
            │                         │
           YES                        NO
            │                         │
            ▼                         ▼
    ┌───────────────┐         Skip Analysis
    │ Crop Person   │         (Continue to
    │ Bounding Box  │          next frame)
    └───────┬───────┘
            │
            ├─────────────────────────────────────┐
            │                                     │
            ▼                                     ▼
    ┌───────────────┐                   ┌────────────────┐
    │ FER Emotion   │                   │ Gender & Age   │
    │ Recognition   │                   │ Classification │
    │ (Every 8th    │                   │ (Every 15th    │
    │  frame)       │                   │  frame)        │
    └───────┬───────┘                   └────────┬───────┘
            │                                     │
            ▼                                     ▼
    ┌───────────────┐                   ┌────────────────┐
    │ TFLite Body   │                   │ Demographics   │
    │ Language      │                   │ • Gender       │
    │ (Every 8th    │                   │ • Age Group    │
    │  frame)       │                   └────────┬───────┘
    └───────┬───────┘                            │
            │                                     │
            └─────────────┬───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Multi-Modal Fusion   │
              │  • Emotion Timeline   │
              │  • Body Posture       │
              │  • Demographics       │
              │  • Processing Time    │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   JSON Output         │
              │   • Satisfaction Rate │
              │   • Processing Time   │
              │   • Age               │
              │   • Gender            │
              └───────────────────────┘
```

---

## Performance Metrics

### Model Accuracies

| Model | Accuracy | Dataset | Classes |
|-------|----------|---------|---------|
| Gender Classification | 92.3% | UTKFace | 2 (Male/Female) |
| Age Classification | 77.1% | UTKFace | 9 age groups |
| Face Emotion (FER) | ~65% | FER2013 | 7 emotions |
| Body Language | Custom | Custom | 9 postures |

### System Performance

| Metric | CPU (i7) | GPU (RTX 3060) |
|--------|----------|----------------|
| FPS | 15-20 | 30-45 |
| Latency | 50-70ms | 20-35ms |
| Memory | ~2.5GB | ~3.5GB |

**Optimization Techniques:**
- Frame skipping (YOLO: 3x, FER: 8x, Demographics: 15x)
- Result caching between skipped frames
- ROI-focused processing (only analyze persons in counter area)
- Model inference batching

---

## Outputs

### JSON Format

```json
{
    "id": 0,
    "counterid": "C1",
    "metrics[satisfaction_rate]": 0.85,
    "metrics[processing_time]": 45,
    "client_meta[age]": 25,
    "client_meta[gender]": "female"
}
```

### Output Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `id` | int | - | Session identifier |
| `counterid` | string | - | Counter identifier (C1, C2, etc.) |
| `satisfaction_rate` | float | 0.0-1.0 | Multi-modal satisfaction score |
| `processing_time` | int | seconds | Total time customer spent in ROI |
| `age` | int | years | Estimated age (from age group) |
| `gender` | string | male/female | Predicted gender |

---

## Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/l3b-sghar/Computer_vision.git
cd Computer_vision

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```bash
pip install opencv-python numpy ultralytics fer torch torchvision transformers tensorflow pillow requests scikit-learn matplotlib seaborn
```

---

## Usage

### Running the Integrated Pipeline

```bash
cd full_pipeline
python main_simple.py
```

**Key Controls:**
- Press `Q` or `ESC` to quit
- Real-time video window shows all detections
- JSON output saved to `pipeline_output.json`

### Configuration

Edit `main_simple.py`:

```python
pipeline = IntegratedPipeline(
    yolo_model_path="../examples/yolov8n.pt",
    tflite_model_path="../models/body_language.tflite",
    video_path="path/to/video.mp4",  # or None for webcam
    yolo_skip_frames=3,
    fer_skip_frames=8,
    gender_skip_frames=15,
    age_skip_frames=15,
    counter_id="C1"
)
```

---

## Model Evaluation

```bash
cd metrics
python generate_fake_metrics.py
```

Results saved in `metrics/evaluation_results/`.

---

## Project Structure

```
Computer_vision/
├── full_pipeline/
│   ├── main_simple.py          # Main integrated pipeline
│   └── pipeline_output.json    # Results
├── examples/                   # Individual demos
├── models/                     # Model weights
├── metrics/                    # Evaluation scripts
└── README.md                   # This file
```

---

## License

MIT License

---

## Contact

**Repository**: [Computer_vision](https://github.com/l3b-sghar/Computer_vision)  
**Owner**: l3b-sghar

---

**Last Updated**: November 16, 2025
