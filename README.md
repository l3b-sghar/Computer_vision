# CBUS — Minimal Computer Vision Utility System

A minimal, environment-agnostic computer-vision prototype that estimates customer satisfaction and interaction processing time using a single camera. CBUS adapts its vision pipeline depending on camera viewpoint and visible body regions (face, upper body, or full body) and fuses temporal context across steps.

---

## Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
  - [Adaptive Vision Pipeline](#adaptive-vision-pipeline)
  - [Facial Emotion Recognition (FER)](#facial-emotion-recognition-fer)
  - [Body Language Interpretation](#body-language-interpretation)
  - [Processing Time Tracking](#processing-time-tracking)
  - [Temporal Task Dependency](#temporal-task-dependency)
- [Outputs](#outputs)
  - [Required Outputs](#required-outputs)
  - [Suggested Add-On Outputs](#suggested-add-on-outputs)
- [Inputs](#inputs)
- [System Flow](#system-flow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration & Customization](#configuration--customization)
- [Notes & Limitations](#notes--limitations)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

CBUS (Computer Vision Utility System) runs a sequential pipeline on frames from a single fixed RGB camera to compute:

- Customer Satisfaction Score (based on facial expressions, body language, and temporal smoothing)
- Processing Time (duration of an interaction)

The system automatically selects an appropriate vision stack (e.g., YOLO, MediaPipe FaceMesh/Holistic, or minimal handcrafted extraction) according to camera angle, distance, and visible regions.

---

## Core Features

### Adaptive Vision Pipeline
Automatically selects between:
- YOLO
- MediaPipe FaceMesh
- MediaPipe Holistic
- Minimal handcrafted extraction (fallback)

Selection criteria:
- Camera angle
- Distance
- Visibility of facial detail and body posture

### Facial Emotion Recognition (FER)
- Detects a minimal set of expressions:
  - Positive (happy / pleased)
  - Neutral
  - Negative (angry / frustrated / sad)

Note: classical FER datasets (FER2013, RAF-DB, AffectNet) do not include "Interested". You can approximate "interest" via attention direction, eyebrow raise, head tilt, and focused gaze.

### Body Language Interpretation
If torso or whole body is available, extracts optional cues:
- Leaning forward / backward
- Hand agitation
- Head nods / shakes
- Shoulder tension
- Posture openness

### Processing Time Tracking
- Starts when a customer enters frame
- Stops when they exit or turn away
- Produces:
  - Total interaction duration
  - (Optional) Active listening duration

### Temporal Task Dependency
- Later stages use context from previous stages:
  - Body posture helps interpret facial expression
  - Emotion timeline influences the final satisfaction score
  - Processing time helps weigh early vs. late emotions

---

## Outputs

### Required Outputs

| Output                  | Description |
|------------------------:|-------------|
| Customer_Satisfaction   | Final score (0–1 or 0–100) combining FER + body language + temporal smoothing |
| Processing_Time         | Time elapsed between interaction start and end |

### Suggested Add-On Outputs
- Engagement Level (low / medium / high)
- Stress / Calmness Index
- Attention Score (gaze + head orientation)
- Body Language Tension Score
- Emotion Timeline Curve
- Confidence Score for predictions
- Interaction Event Log (timestamps for key behaviors)

---

## Inputs

| Input             | Source            | Notes |
|------------------:|-------------------|-------|
| RGB Camera Feed   | Single fixed camera | May include face, torso, or full body depending on installation |
| Previous Task History | Internal memory | Ensures current inference uses earlier steps’ results |

---

## System Flow

1. Capture frame
2. Decide pipeline → YOLO / MediaPipe / fallback
3. Extract keypoints (face or body)
4. Perform FER (minimal emotion set)
5. Estimate body-language cues
6. Fuse multi-step results (temporal smoothing)
7. Compute satisfaction score
8. Track total interaction time
9. Output final metrics

---

## Project Structure

```
├── README.md
├── src/
│   ├── pipeline_selector.py
│   ├── face_emotion_model.py
│   ├── body_language_analysis.py
│   ├── satisfaction_fusion.py
│   ├── time_tracker.py
│   └── utils/
│       └── preprocessing.py
├── config/
│   └── system_config.yaml
└── demo/
    └── example_video.mp4
```

---

## Installation

```bash
git clone <repo-url>
cd Computer_vision
pip install -r requirements.txt
```

---

## Usage

Run with the default camera (0):

```bash
python src/main.py --camera 0
```

Adjust flags/config as needed. See `config/system_config.yaml` for tunable parameters.

---

## Configuration & Customization

Edit `config/system_config.yaml` to tune:
- Emotion sensitivity
- Body language thresholds
- Camera distance presets
- Output smoothing parameters

Consider adding profiles for different camera placements (e.g., face-only, torso, full-body).

---

## Notes & Limitations

- Classical FER cannot perfectly detect "interest"; use custom heuristics combining gaze, head pose, and eyebrow motion.
- Camera angle and distance strongly affect performance.
- Designed as a prototype/hackathon demonstrator and requires calibration and testing before production use.

---

## Contributing

Contributions are welcome. Suggested workflow:
1. Create a branch for your change.
2. Add tests and update docs where applicable.
3. Open a PR describing the change.

Please follow repository coding standards and keep changes focused.

---

## License

MIT License

---

## Contact

Repository owner: l3b-sghar

For questions, issues, or feature requests, please open an issue in the repo.
