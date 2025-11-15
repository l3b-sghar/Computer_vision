# Project Structure Documentation

This document describes the structure of the emotion and body-language analyzer project.

## Directory Structure

```
Computer_vision/
├── main.py                    # Main entry point
├── config.py                  # Configuration management
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
│
├── detectors/                 # Detection modules
│   ├── __init__.py           # Package initialization with DetectorProtocol
│   ├── base_detector.py      # Abstract base class for detectors
│   ├── yolo_detector.py      # YOLO detector implementation (placeholder)
│   └── mediapipe_detector.py # MediaPipe detector implementation (placeholder)
│
├── analytics/                 # Analytics modules
│   ├── __init__.py           # Package initialization with AnalyticsResult
│   ├── emotion_analyzer.py   # Emotion recognition (placeholder)
│   ├── body_language_analyzer.py  # Body language interpretation (placeholder)
│   └── satisfaction_scorer.py     # Satisfaction score computation (placeholder)
│
└── utils/                     # Utility modules
    ├── __init__.py           # Package initialization with common functions
    ├── video_processor.py    # Video input/output handling
    └── logger.py             # Logging utilities
```

## Core Components

### 1. Main Entry Point (`main.py`)

The main entry point provides:
- Command-line interface for running the analyzer
- `EmotionAnalyzer` class that coordinates the pipeline
- Video processing loop with visualization
- Result generation and display

**Usage:**
```bash
python main.py                    # Use default webcam
python main.py --source 0         # Use webcam 0
python main.py --source video.mp4 # Use video file
python main.py --detector yolo    # Force YOLO detector
python main.py --no-display       # Disable visualization
```

### 2. Configuration (`config.py`)

Centralized configuration management:
- Detector settings (YOLO, MediaPipe)
- Video input/output settings
- Emotion recognition parameters
- Analytics weights and thresholds
- Optional scores configuration

Configuration can be set via:
- Environment variables
- Direct class attribute modification
- Dictionary-based updates

### 3. Detectors Package (`detectors/`)

Provides a flexible interface for different detection backends:

**`base_detector.py`**: Abstract base class defining the detector interface
- `initialize()`: Set up detector resources
- `detect(frame)`: Perform detection on a frame
- `is_available()`: Check if detector is ready
- `cleanup()`: Release resources

**`yolo_detector.py`**: YOLO-based detector (to be implemented)
- Uses Ultralytics YOLO for object and pose detection
- Suitable for distant cameras or full-body views

**`mediapipe_detector.py`**: MediaPipe-based detector (to be implemented)
- Uses MediaPipe Holistic for face, pose, and hand tracking
- Suitable for close-up views with detailed facial features

### 4. Analytics Package (`analytics/`)

Provides modules for analyzing detected features:

**`emotion_analyzer.py`**: Emotion recognition
- Analyzes facial expressions
- Tracks emotion history
- Computes temporal emotion patterns

**`body_language_analyzer.py`**: Body language interpretation
- Analyzes posture and gestures
- Detects body language events
- Tracks movement patterns

**`satisfaction_scorer.py`**: Satisfaction score computation
- Fuses emotion and body language signals
- Computes temporal satisfaction scores
- Generates optional scores (attention, stress, hesitancy)

**`AnalyticsResult`**: Data class for results with required and optional outputs:
- Required: `customer_satisfaction_score`, `processing_time_seconds`
- Optional: `attention_score`, `stress_score`, `hesitancy_score`, `body_language_events`

### 5. Utils Package (`utils/`)

Common utility functions:

**Core utilities** (`__init__.py`):
- `load_video_source()`: Open video source (webcam or file)
- `preprocess_frame()`: Resize and normalize frames
- `draw_bbox()`: Draw bounding boxes with labels
- `calculate_fps()`: Compute current FPS

**`video_processor.py`**: Video I/O handling
- `VideoProcessor` class for reading/writing video
- Frame generator for streaming processing
- Context manager support

**`logger.py`**: Logging setup
- Console and file logging
- Configurable log levels
- Timestamp formatting

## Design Principles

### 1. Flexibility

The structure supports multiple detector backends (YOLO, MediaPipe) through:
- Abstract base class (`BaseDetector`)
- Protocol definition (`DetectorProtocol`)
- Configuration-based selection

### 2. Modularity

Each component has a clear responsibility:
- Detectors: Find faces and poses
- Analytics: Interpret detected features
- Utils: Provide common functionality
- Config: Centralize settings

### 3. Extensibility

Future tasks can easily add:
- New detector implementations
- Additional analytics modules
- Custom scoring algorithms
- New utility functions

### 4. Stability

Stable interfaces ensure future tasks won't break existing code:
- Well-defined protocols and base classes
- Consistent return types
- Clear module boundaries
- Comprehensive docstrings

## Output Specification

### Required Outputs

1. **customer_satisfaction_score** (float, 0-1)
   - Overall satisfaction based on emotion and body language
   - Computed by `SatisfactionScorer`

2. **processing_time_seconds** (float)
   - Duration of the interaction
   - Tracked from frame timestamps

### Optional Outputs

3. **attention_score** (float, 0-1)
   - Based on gaze direction and head pose
   - Enabled via `Config.ENABLE_ATTENTION_SCORE`

4. **stress_score** (float, 0-1)
   - Based on facial tension and body language
   - Enabled via `Config.ENABLE_STRESS_SCORE`

5. **hesitancy_score** (float, 0-1)
   - Based on behavioral patterns
   - Enabled via `Config.ENABLE_HESITANCY_SCORE`

6. **body_language_events** (list)
   - Timestamped body language events
   - Enabled via `Config.ENABLE_BODY_LANGUAGE_EVENTS`

## Dependencies

Core dependencies in `requirements.txt`:
- **OpenCV**: Video I/O and image processing
- **NumPy**: Numerical operations
- **TensorFlow/PyTorch**: Deep learning frameworks
- **Ultralytics**: YOLO implementation
- **MediaPipe**: Google's ML solutions
- **scikit-learn/pandas**: Data processing
- **matplotlib/seaborn**: Visualization

## Future Development

This structure is designed to support the following upcoming tasks:

1. **Detector Implementation**
   - Implement YOLO detector
   - Implement MediaPipe detector
   - Add automatic detector selection

2. **Emotion Recognition**
   - Train/load emotion recognition model
   - Implement facial feature extraction
   - Add temporal smoothing

3. **Body Language Analysis**
   - Implement posture analysis
   - Add gesture detection
   - Track movement patterns

4. **Satisfaction Scoring**
   - Implement fusion algorithm
   - Add temporal weighting
   - Compute optional scores

5. **Testing and Validation**
   - Add unit tests
   - Add integration tests
   - Validate on real videos

## Notes

- All placeholder implementations are marked with "to be implemented in future tasks"
- The structure follows Python best practices (PEP 8)
- Module docstrings follow Google style
- Type hints are used throughout for better IDE support
- The `.gitignore` excludes common Python artifacts and project-specific files
