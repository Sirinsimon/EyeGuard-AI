# Quick Start Guide

## You're Ready to Run! 🎉

All dependencies are installed and the model file is downloaded. Just run:

```bash
python sleep_detector.py
```

## What to Expect

1. A window will open showing your webcam feed
2. The system will detect your face and draw landmarks
3. Your eyes will be highlighted with yellow circles
4. Status will show at the top:
   - **AWAKE** (Green) - Normal state
   - **SLEEPING** (Red) - Eyes closed for 3+ seconds
5. You'll see:
   - Current EAR (Eye Aspect Ratio) value
   - Total sleep time accumulated
6. An alarm will beep when sleep is detected

## Controls

- **ESC** - Exit the application

## Tips for Best Results

1. **Lighting**: Ensure your face is well-lit
2. **Distance**: Sit at a comfortable distance from the camera (arm's length)
3. **Position**: Keep your face centered in the frame
4. **Angle**: Face the camera directly for best detection

## Adjusting Sensitivity

Edit `sleep_detector.py` and modify these values:

```python
EAR_THRESHOLD = 0.25           # Lower = more sensitive (0.20-0.30 recommended)
SLEEP_TIME_THRESHOLD = 3.0     # Seconds before marking as sleeping
```

## Troubleshooting

**"No Face Detected"**
- Check lighting
- Move closer to camera
- Ensure camera is working

**High/Low sensitivity**
- Adjust `EAR_THRESHOLD` value
- Lower value = more sensitive to eye closure
- Higher value = less sensitive

**Camera not working**
- Close other apps using the camera
- Check camera permissions in Windows Settings

## Files in This Project

- `sleep_detector.py` - Main application
- `face_landmarker.task` - MediaPipe model file (3.6 MB)
- `requirements.txt` - Python dependencies
- `README.md` - Full documentation
- `QUICKSTART.md` - This file

Enjoy your Sleep Detection System! 😴👁️
