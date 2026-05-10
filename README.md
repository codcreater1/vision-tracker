██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ████████╗██████╗  █████╗  ██████╗██╗  ██╗███████╗██████╗
██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██║ ██╔╝██╔════╝██╔══██╗
██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║       ██║   ██████╔╝███████║██║     █████╔╝ █████╗  ██████╔╝
╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║       ██║   ██╔══██╗██╔══██║██║     ██╔═██╗ ██╔══╝  ██╔══██╗
 ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║       ██║   ██║  ██║██║  ██║╚██████╗██║  ██╗███████╗██║  ██║
  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
Real-Time Face Mesh + Hand Tracking
478 facial landmarks · dual hand skeleton · gesture recognition · blink detection

Turn your webcam into a real-time computer vision lab.
No cloud. No server. Just Python, MediaPipe and OpenCV — running entirely on your machine.

<br>

</div>
What Is This?
Vision Tracker is a production-quality computer vision application that detects and visualises your face and hands in real time using only a standard webcam.
It uses Google's MediaPipe Tasks API (the latest 0.10+ architecture) — not the deprecated mp.solutions — with two dedicated neural network models running in parallel:

A Face Landmarker that tracks 478 precise points across your face, including your irises
A Hand Landmarker that tracks 21 skeletal joints per hand for both hands simultaneously

Everything runs locally. No data leaves your device.

Feature Overview
┌─────────────────────────────────────────────────────────────┐
│                     VISION TRACKER                          │
├──────────────────────┬──────────────────────────────────────┤
│  FACE ANALYSIS       │  HAND ANALYSIS                       │
│                      │                                      │
│  478-point mesh      │  21 joints per hand                  │
│  Iris ring tracking  │  Both hands simultaneously           │
│  Eye contours        │  Right: orange  /  Left: violet      │
│  Face oval outline   │  Fingertip highlight (larger dots)   │
│  Blink detection     │  Gesture recognition                 │
│    via EAR algo      │    Fist / Peace / Thumbs Up          │
│    L + R eye         │    Pointing / Open Hand              │
├──────────────────────┴──────────────────────────────────────┤
│  LIVE HUD                                                   │
│                                                             │
│  Real-time FPS (30-frame rolling average)                   │
│  Detection status labels                                    │
│  Per-hand gesture name                                      │
│  Blink state (open / BLINK) per eye                         │
│  Overlay opacity level + mirror state                       │
├─────────────────────────────────────────────────────────────┤
│  CONTROLS                                                   │
│                                                             │
│  Q  quit          S  screenshot      M  mirror toggle       │
│  +  opacity up    -  opacity down                           │
└─────────────────────────────────────────────────────────────┘

Quick Start
Prerequisites

Python 3.10 or higher
A working webcam
Internet connection (for first-run model download only)

Install
bashgit clone https://github.com/codcreater1/vision-tracker.git
cd vision-tracker
pip install opencv-python mediapipe numpy
Run
bashpython vision_tracker.py
On first launch the app automatically downloads two neural network model files (~30 MB total) into the project folder. This happens once and never again.

Keyboard Controls
KeyActionQQuit the applicationSSave a screenshot to the screenshots/ folderMToggle mirror mode (horizontal flip)+Increase mesh overlay opacity-Decrease mesh overlay opacity

Architecture
                        ┌──────────────┐
                        │   Webcam     │
                        └──────┬───────┘
                               │  BGR frame
                               ▼
                        ┌──────────────┐
                        │  BGR → RGB   │
                        │  mp.Image()  │
                        └──────┬───────┘
                               │
               ┌───────────────┴───────────────┐
               │                               │
               ▼                               ▼
   ┌───────────────────────┐     ┌─────────────────────────┐
   │   FaceLandmarker      │     │    HandLandmarker        │
   │   478 landmarks       │     │    21 landmarks x hand   │
   └──────────┬────────────┘     └────────────┬────────────┘
              │                               │
     ┌────────┴────────┐             ┌────────┴────────┐
     │                 │             │                 │
     ▼                 ▼             ▼                 ▼
  draw_face()    eye_aspect     draw_hand()     classify_gesture()
  mesh dots      ratio (EAR)    skeleton        fingers_up()
  oval + iris    blink detect   colored joints  Fist/Peace/etc.
     │                 │             │                 │
     └────────┬────────┘             └────────┬────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  blend_overlay() │
                    │  alpha composite │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   draw_hud()     │
                    │   FPS / stats    │
                    └────────┬─────────┘
                             │
                             ▼
                      cv2.imshow()

Blink Detection — EAR Algorithm
The app detects eye blinks using the Eye Aspect Ratio (EAR), a well-established technique from facial landmark research:
      p2    p3
       *----*
  p1 *        * p4      EAR = ( ||p2-p6|| + ||p3-p5|| )
       *----*                  ─────────────────────────
      p6    p5                      2 * ||p1-p4||
When EAR < 0.20, the eye is considered closed.
The HUD shows BLINK / open independently for each eye.

Gesture Recognition
The app classifies hand gestures by checking which fingers are extended:
GestureFingers UpLabelClosed fistnoneFistAll fingersall fiveOpen HandIndex + Middle2PeaceThumb onlythumbThumbs UpIndex onlyindexPointingPinky onlypinkyPinkyOthermixedCustom (n fingers)

Configuration
All visual and detection parameters are centralised in the Config class — no magic numbers scattered through the code:
pythonclass Config:
    # Detection sensitivity
    MIN_DETECTION_CONF = 0.55   # higher = fewer false positives
    MIN_TRACKING_CONF  = 0.50   # lower  = handles fast movement better
    MAX_FACES          = 2
    MAX_HANDS          = 2

    # Color palette (BGR format)
    FACE_DOT_COLOR     = (0,   230, 140)   # green-mint
    FACE_CONTOUR_COLOR = (255, 220,  60)   # amber
    FACE_IRIS_COLOR    = (100, 200, 255)   # sky-blue
    HAND_R_DOT         = (0,   160, 255)   # orange  (right hand)
    HAND_L_DOT         = (220,  60, 255)   # violet  (left hand)

    # Performance
    FPS_WINDOW         = 30    # rolling average window size
    SCREENSHOT_DIR     = "screenshots"

Project Structure
vision-tracker/
│
├── vision_tracker.py       # Main application (single-file, self-contained)
│
├── face_landmarker.task    # Downloaded automatically on first run
├── hand_landmarker.task    # Downloaded automatically on first run
│
├── screenshots/            # Created automatically when you press S
│   └── capture_YYYYMMDD_HHMMSS.png
│
└── README.md

Requirements
PackageVersionPython>= 3.10opencv-python>= 4.8mediapipe>= 0.10numpy>= 1.24

Why MediaPipe Tasks API?
The old mp.solutions.holistic API was removed in MediaPipe 0.10. This project uses the new Tasks API (mediapipe.tasks.python.vision) which offers:

Separate, composable landmarker models
RunningMode.VIDEO for efficient per-frame inference with timestamps
Cleaner, more maintainable code with explicit model loading


License
MIT License — free to use, modify and distribute.
Built by @codcreater1

<div align="center">
If this project helped you, consider leaving a star!
</div>
