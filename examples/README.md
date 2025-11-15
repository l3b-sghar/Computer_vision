# Examples

This directory contains example scripts demonstrating the usage of various modules in the Computer Vision system.

## FER Demo (`fer_demo.py`)

Comprehensive demonstration of the Facial Emotion Recognition (FER) module.

### Running the Demo

```bash
python examples/fer_demo.py
```

### What it demonstrates

1. **Basic FER Usage**: Initialize and use the FER module
2. **Multiple Input Formats**: Different ways to provide face data
3. **Temporal Smoothing**: How to use temporal smoothing for stable predictions
4. **Detector Integration**: Integration with detector outputs (MediaPipe, YOLO)
5. **Fallback Behavior**: How FER handles missing face data
6. **Satisfaction Mapping**: Emotion to satisfaction score mapping

### Requirements

- TensorFlow (for FER model)
- NumPy
- OpenCV

Install requirements:
```bash
pip install tensorflow numpy opencv-python
```

### Output

The demo will show emotion recognition results for various input scenarios, including:
- Detected emotions and confidence scores
- Satisfaction scores based on emotions
- Temporal smoothing effects
- Integration with detector outputs
- Fallback behavior when no face is detected
