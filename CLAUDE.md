# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VisionLens AR is a real-time virtual try-on system built with MediaPipe and OpenCV. The application detects faces and overlays virtual accessories (glasses, beards) with accurate 3D positioning and natural blending effects.

## Core Architecture

The application follows a modular architecture with separate concerns:

- **main.py**: Application entry point handling camera capture, user input, and the main render loop
- **visionlens/**: Core modules containing the AR processing logic
  - **face_detector.py**: MediaPipe-based face detection with 468 3D landmarks, GPU/CPU optimization
  - **accessory_manager.py**: Asset loading, caching, and rendering orchestration
  - **transform_utils.py**: 3D transformation calculations for positioning accessories
  - **effects.py**: Alpha blending and multi-accessory compositing
  - **model_downloader.py**: Automatic MediaPipe model file management

## Development Commands

### Running the Application
```bash
python main.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Testing
The project doesn't include automated tests. Manual testing involves:
1. Running the application and verifying camera access
2. Testing accessory switching with 'g' and 'b' keys
3. Verifying smooth 30+ FPS performance on CPU

## Key Technical Details

### Performance Optimization
- Frame detection throttling: Process face detection every N frames (configurable in main.py:59)
- Automatic GPU detection and fallback to CPU
- Image caching for accessories to avoid repeated file I/O
- Configurable camera resolution (640x480 default)

### Face Detection System
- Uses MediaPipe Face Landmarker with 468 3D points
- Key landmark indices: left eye center (159), right eye center (386), nose bridge (6), chin (175)
- Supports both GPU and CPU inference with automatic detection

### 3D Transformation Pipeline
Accessories are positioned using:
1. **Scale calculation**: Based on inter-ocular distance and face measurements
2. **Rotation calculation**: Head pose estimation (Pitch, Yaw, Roll) from landmarks
3. **Perspective transform**: Non-linear distortion based on head rotation for realistic depth

### Asset Management
- PNG files with alpha channels for transparency
- Automatic discovery and loading from assets/ subdirectories
- In-memory caching for performance
- Center-point positioning system for accurate alignment

### Rendering Pipeline
```
Camera Frame → MediaPipe Detection → Landmark Extraction → Transform Calculation →
Asset Loading → Apply Transform → Alpha Blending → Output Window
```

## File Structure Requirements

```
assets/
├── glasses/           # PNG files with transparency
└── beard/            # PNG files with transparency

models/                # Auto-created, contains MediaPipe model files
```

## Model Files

The application automatically downloads `face_landmarker.task` on first run. If automatic download fails:
- Download from: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
- Save to: `./models/face_landmarker.task`

## Controls

- **'g'**: Cycle through glasses styles
- **'b'**: Cycle through beard styles
- **'q'**: Exit application

## Development Notes

- The application uses horizontal frame flipping (mirror effect) for natural user experience
- Face detection results are cached between frames for performance
- All coordinates use MediaPipe's 468-point landmark system
- Alpha blending ensures natural edge transitions between accessories and skin
- Multi-accessory rendering supported (glasses + beard simultaneously)