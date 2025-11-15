# Examples

This directory contains example scripts demonstrating the usage of various modules in the Computer Vision system.

## Live Camera FER Demo (`fer_demo.py`)

Real-time facial emotion recognition using your webcam with accurate pre-trained models.

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

### Output Example

When working correctly with DeepFace, you'll see:
- Colored boxes around detected faces
- Emotion labels (e.g., "HAPPY (0.89)")
- Satisfaction score (e.g., "Satisfaction: 0.92")
- Probability bars for all 7 emotions
- Real-time FPS and face count
