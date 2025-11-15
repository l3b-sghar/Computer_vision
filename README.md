📌 Overview

This repository contains the full prototype specification for CBUS, a minimal and environment-agnostic computer-vision system that estimates:

Customer Satisfaction Score (via Face Expression Recognition & body language cues)

Processing Time (duration of the customer request interaction)

The system uses only one camera, which may view:

The customer’s face,

The customer’s upper body, or

The entire body, depending on installation.

All tasks are executed in sequence, with each step taking into account the result of the previous step (temporal context).

The system automatically selects the best vision stack (e.g., YOLO, Mediapipe Holistic, etc.) depending on the camera viewpoint and the visible body region.

✨ Core Features
1. Adaptive Vision Pipeline

Automatically selects between YOLO, Mediapipe FaceMesh, Mediapipe Holistic, or minimal handcrafted extraction.

Chooses the optimal method based on:

Camera angle

Distance

Visibility of facial details or body posture

2. Facial Emotion Recognition (FER)

Detects a minimal set of expressions:

Positive (happy/pleased)

Neutral

Negative (angry/frustrated/sad)

❓ Does classical FER include “Interested”?
No. Traditional FER datasets (FER2013, RAF-DB, AffectNet) do NOT include Interest as an emotion category.
You may approximate “interest” using combinations of:

attention direction,

eyebrow raise,

head tilt,

focused gaze.

3. Body Language Interpretation

If the whole body or torso is visible, optional cues are extracted:

Leaning forward/backward

Hand agitation

Head nods/shakes

Shoulder tension

Posture openness

4. Processing Time Tracking

Starts when the customer enters frame

Stops when they exit or turn away

Produces:

Total interaction duration

Active listening duration (optional)

5. Temporal Task Dependency

Every stage uses context from the previous ones:

Body posture → helps interpret facial expressions

Emotion timeline → influences final satisfaction score

Processing time → helps weigh early vs. late emotions

📤 Outputs
Required Outputs
Output	Description
Customer_Satisfaction	Final score from 0–1 or 0–100, combining FER + body language + temporal smoothing.
Processing_Time	Time elapsed between interaction start & end.
Suggested Add-On Outputs

Optional extendable outputs:

Engagement Level (low / medium / high)

Stress / Calmness Index

Attention Score (based on gaze + head orientation)

Body Language Tension Score

Emotion Timeline Curve

Confidence Score for predictions

Interaction Event Log (timestamps for key behaviors)

📥 Inputs
Input	Source	Notes
RGB Camera Feed	Single fixed camera	Deals with face, torso, or full body depending on installation.
Previous Task History	Internal memory	Ensures current inference uses earlier steps’ results.
🧠 System Flow
1. Capture frame
2. Decide pipeline → YOLO / Mediapipe / fallback
3. Extract keypoints (face or body)
4. Perform FER (minimal emotion set)
5. Estimate body-language cues
6. Fuse multi-step results (temporal smoothing)
7. Compute satisfaction score
8. Track total interaction time
9. Output final metrics

📦 Project Structure
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

🚀 Installation
git clone <repo-url>
cd CBUS
pip install -r requirements.txt

▶️ Running the System
python src/main.py --camera 0

🛠️ Customization

Edit config/system_config.yaml to tune:

Emotion sensitivity

Body language thresholds

Camera distance presets

Output smoothing parameters

📚 Notes & Limitations

Classical FER cannot perfectly detect interest → may need custom heuristics.

Camera angle strongly affects performance.

Designed for hackathon-scale demonstrators; not intended for production without calibration.

📄 License

MIT License
