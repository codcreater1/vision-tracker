<div align="center">

<h1>Vision Tracker</h1>

<p><strong>Real-time Face Mesh + Hand Tracking with MediaPipe & OpenCV</strong></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/MediaPipe-0.10%2B-FF6F00?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
</p>

<p><em>Turn your webcam into a real-time computer vision lab.<br>No cloud. No server. Runs entirely on your machine.</em></p>

</div>

---

## Overview

**Vision Tracker** detects and visualises your face and hands in real time using only a standard webcam.

It uses **Google's MediaPipe Tasks API** (0.10+ architecture) with two neural network models running in parallel:

- **Face Landmarker** — tracks 478 precise points across your face, including irises
- **Hand Landmarker** — tracks 21 skeletal joints per hand, both hands simultaneously

---

## Features

### Face Analysis
- 478-point mesh drawn across the entire face
- Iris ring detection (left & right eye separately)
- Eye contour highlighting
- Face oval outline
- **Blink detection** via Eye Aspect Ratio (EAR) algorithm — shown live in HUD

### Hand Analysis
- Both hands tracked simultaneously
- 21 landmarks per hand with colored joints
- Right hand: orange | Left hand: violet
- Fingertip dots displayed larger for clarity
- **Gesture recognition**: Fist, Peace, Thumbs Up, Pointing, Open Hand

### Live HUD
- Real-time FPS counter (30-frame rolling average)
- Detection status per face and hand
- Per-hand gesture label
- Blink state (open / BLINK) for each eye independently
- Overlay opacity level and mirror state display

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/codcreater1/vision-tracker.git
cd vision-tracker
```

### 2. Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

### 3. Run

```bash
python vision_tracker.py
```

> On first launch, two MediaPipe model files (~30 MB total) are downloaded automatically. This only happens once.

---

## Keyboard Controls

| Key | Action |
|:---:|--------|
| `Q` | Quit the application |
| `S` | Save screenshot to `screenshots/` folder |
| `M` | Toggle mirror mode (horizontal flip) |
| `+` | Increase mesh overlay opacity |
| `-` | Decrease mesh overlay opacity |

---

## How It Works

```
Webcam Frame
     |
     v
BGR -> RGB -> mp.Image
     |
     +--------> FaceLandmarker
     |               478 landmarks
     |               - Face mesh dots
     |               - Oval + eye contours
     |               - Iris rings
     |               - Blink detection (EAR)
     |
     +--------> HandLandmarker
                     21 landmarks x hand
                     - Skeleton drawing
                     - Finger state detection
                     - Gesture classification
     |
     v
Alpha blend overlay onto frame
     |
     v
Draw HUD (FPS, labels, gestures)
     |
     v
cv2.imshow()
```

---

## Blink Detection — EAR Algorithm

Blinks are detected using the **Eye Aspect Ratio (EAR)**, a technique from facial landmark research:

```
EAR = ( ||p2-p6|| + ||p3-p5|| ) / ( 2 * ||p1-p4|| )
```

When `EAR < 0.20`, the eye is classified as **closed**.
The HUD displays blink state independently for left and right eye.

---

## Gesture Recognition

| Gesture | Condition |
|---------|-----------|
| Fist | No fingers extended |
| Open Hand | All five fingers extended |
| Peace | Index + Middle extended |
| Thumbs Up | Thumb only extended |
| Pointing | Index finger only |
| Pinky | Pinky finger only |
| Custom | Any other combination |

---

## Configuration

All parameters are in the `Config` class at the top of `vision_tracker.py`:

```python
class Config:
    MIN_DETECTION_CONF = 0.55   # Detection sensitivity
    MIN_TRACKING_CONF  = 0.50   # Tracking sensitivity
    MAX_FACES          = 2      # Max simultaneous faces
    MAX_HANDS          = 2      # Max simultaneous hands

    FACE_DOT_COLOR     = (0,   230, 140)  # Green-mint
    FACE_CONTOUR_COLOR = (255, 220,  60)  # Amber
    FACE_IRIS_COLOR    = (100, 200, 255)  # Sky-blue
    HAND_R_DOT         = (0,   160, 255)  # Orange (right hand)
    HAND_L_DOT         = (220,  60, 255)  # Violet (left hand)

    FPS_WINDOW         = 30               # Rolling average window
    SCREENSHOT_DIR     = "screenshots"
```

---

## Project Structure

```
vision-tracker/
├── vision_tracker.py       # Main application
├── face_landmarker.task    # Auto-downloaded on first run
├── hand_landmarker.task    # Auto-downloaded on first run
├── screenshots/            # Saved captures (created automatically)
└── README.md
```

---

## Requirements

| Package | Version |
|---------|---------|
| Python | >= 3.10 |
| opencv-python | >= 4.8 |
| mediapipe | >= 0.10 |
| numpy | >= 1.24 |

---

## Why MediaPipe Tasks API?

The old `mp.solutions.holistic` was removed in MediaPipe 0.10. This project uses the modern **Tasks API** which provides:

- Separate, composable landmarker models
- `RunningMode.VIDEO` for efficient per-frame inference
- Cleaner architecture with explicit model loading

---

## License

MIT License — free to use, modify and distribute.

Built by [@codcreater1](https://github.com/codcreater1)

---

<div align="center">
<sub>If this project was useful, consider leaving a star!</sub>
</div>
