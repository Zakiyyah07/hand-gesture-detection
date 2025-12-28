# hand-gesture-detection
A simple, experimental hand gesture recognition system built with Python. This project leverages computer vision to detect hand gestures, convert them into text or speech (TTS), and features a built-in training mode for custom gesture creation.

Note: As this is my first attempt, you might find some "messy" code or bugs. I apologize for any issues I'm still learning and trying my best to improve!

Features
1. Multi-Mode Detection: Seamlessly switch between:

  -Text Mode: Recognize common phrases and names.

  -Number Mode: Accurate detection for numbers 0–10.

  -Pose Mode: Detect dynamic movements like waving.

2. Gesture Training: Record and save your own custom gestures with personalized text associations.

3. Visual Feedback: Real-time camera feed with hand landmark drawings, debug information, and on-screen subtitles.

4. Logging & Counters: Automatically tracks detected gestures and saves history to a log file.

5. Stable Detection: Includes adjustable brightness/contrast sliders and a "cooldown" mechanism to prevent flickering or double-triggering.


How to Use
Once the application is running, the camera will initialize. Use the following keyboard shortcuts to control the system:
## ⌨️ Keyboard Controls
- **t** → Switch to Text Mode (Phrases / Names)
- **n** → Switch to Number Mode (0–10)
- **p** → Switch to Pose Mode (Waving)
- **r** → Toggle Training Mode on / off
- **s** → Save current gesture (Training Mode only)
- **l** → Change TTS Language
- **ESC / q** → Quit application
