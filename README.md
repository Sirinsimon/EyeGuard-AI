# Real-time Sleep Detection System

A Python-based sleep detection system that uses computer vision to monitor eye closure and detect drowsiness in real-time using your laptop webcam.

## Features

- ✅ Real-time face and eye detection using MediaPipe Face Mesh
- ✅ Eye Aspect Ratio (EAR) calculation for accurate eye state detection
- ✅ Timer-based sleep detection (3-second threshold)
- ✅ Visual feedback with color-coded status (Green=AWAKE, Red=SLEEPING)
- ✅ Eye landmarks and face mesh visualization
- ✅ Loud alarm sound to wake up user when sleep is detected
- ✅ Centered video window for better viewing experience
- ✅ Total sleep duration tracking
- ✅ Optimized for real-time performance

## Tech Stack

- Python 3.8+
- OpenCV - Video capture and processing
- MediaPipe - Face mesh and landmark detection
- NumPy - Numerical computations
- SciPy - Distance calculations
- Pygame - Alarm sound

## Installation

### Step 1: Install Python Dependencies

```bash
pip install opencv-python mediapipe numpy scipy pygame screeninfo
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Step 2: Download MediaPipe Face Landmarker Model

Download the face landmarker model file and save it in the same directory as `sleep_detector.py`:

**Direct Download Link:**
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

Or use this command:

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile "face_landmarker.task"
```

**Linux/Mac:**
```bash
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

### Step 3: Run the Application

```bash
python sleep_detector.py
```

## How It Works

1. **Face Detection**: MediaPipe Face Mesh detects faces and extracts 468 facial landmarks
2. **Eye Landmark Extraction**: Specific landmarks for left and right eyes are identified
3. **EAR Calculation**: Eye Aspect Ratio is computed using the formula:
   ```
   EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
   ```
4. **Threshold Comparison**: 
   - EAR < 0.25 → Eyes closed
   - EAR ≥ 0.25 → Eyes open
5. **Sleep Detection**: If eyes remain closed for more than 3 seconds, status changes to "SLEEPING"
6. **Visual Feedback**: Real-time status display with color coding and statistics

## Usage

- The application will automatically access your webcam
- The video window will appear centered on your screen
- Keep your face visible to the camera
- The system will display:
  - Current status (AWAKE/SLEEPING)
  - Eye Aspect Ratio (EAR) value
  - Total sleep time accumulated
  - Eye landmarks and face mesh overlay
- When sleep is detected (eyes closed for 3+ seconds):
  - Status turns RED
  - A loud, pulsing alarm sound plays repeatedly to wake you up
- Press **ESC** to exit the application

## Configuration

You can adjust these parameters in `sleep_detector.py`:

```python
EAR_THRESHOLD = 0.25           # Lower = more sensitive to eye closure
SLEEP_TIME_THRESHOLD = 3.0     # Seconds before marking as sleeping
```

## Code Structure

The code is modular with separate functions:

- `calculate_ear()` - Computes Eye Aspect Ratio
- `extract_eye_landmarks()` - Extracts eye coordinates from face mesh
- `detect_sleep_status()` - Implements timer-based sleep detection logic
- `draw_eye_landmarks()` - Visualizes eye landmarks on frame
- `play_alarm()` - Plays alarm sound when sleep detected
- `main()` - Main loop for video processing

## Troubleshooting

**Camera not accessible:**
- Ensure no other application is using the webcam
- Check camera permissions in your OS settings

**Poor detection accuracy:**
- Ensure good lighting conditions
- Keep face at a reasonable distance from camera
- Adjust EAR_THRESHOLD if needed

**Alarm not working:**
- Pygame might not be properly installed
- The system will continue working without alarm

## Performance

- Optimized for real-time processing
- Uses MediaPipe's efficient face mesh model
- Minimal lag on modern hardware
- Processes at camera frame rate (typically 30 FPS)

## License

MIT License - Feel free to use and modify for your projects.
