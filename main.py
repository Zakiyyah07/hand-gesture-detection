import sys
import json
import os
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import pygame
import threading
import time
import random

# ============================
# VALIDATION
# ============================
MIN_VER = (3, 9)
MAX_VER = (3, 11)
REQUIRED_LIBS = [
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("numpy", "numpy"),
    ("gtts", "gTTS"),
    ("pygame", "pygame"),
]

def validate_python_version():
    current_ver = sys.version_info
    if not (MIN_VER <= (current_ver.major, current_ver.minor) <= MAX_VER):
        print("============================================")
        print("PYTHON VERSION INCOMPATIBLE")
        print("============================================")
        print(f"Current version: Python {current_ver.major}.{current_ver.minor}.{current_ver.micro}")
        print("This script is compatible with Python 3.9 to 3.11 only.")
        print("Please change your Python version and try again.")
        print("============================================")
        sys.exit(1)

def validate_libraries():
    missing = []
    for module, pip_name in REQUIRED_LIBS:
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print("============================================")
        print("MISSING LIBRARIES")
        print("============================================")
        print("Missing libraries:")
        for lib in missing:
            print(f" - {lib}")
        print("\nInstall with:")
        print("pip install " + " ".join(missing))
        print("============================================")
        sys.exit(1)

validate_python_version()
validate_libraries()

# ============================
# CONSTANTS
# ============================
AUTHOR_NAME = "Kiyyaww"
TTS_DIR = "tts_temp_(auto_delete)"
CUSTOM_GESTURES_FILE = "custom_gestures.json"
LOG_FILE = "gesture_log.txt"

GESTURE_TEXTS = {
    # Main phrases
    "hello": "Hello",
    "my_name": "My name is",
    "zakiyyah": "Zakiyyah",
    "kiyya": "Kiyyah",
    "about_me": "I am an Information Systems major\nfrom Hasanuddin University.",
    "nice_meet": "Nice to meet you!",
    "love_you": "Love You All",
    
    # Numbers
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    
    # Basic letters
    "a": "A", "b": "B", "c": "C",
    
    # Poses
    "waving": "Waving hello",
    "standing": "I am standing",
}

COLORS = {
    "subtitle": (60, 173, 3),
    "fps": (0, 255, 0),
    "author": (255, 255, 0),
    "counter": (255, 0, 255),
    "debug": (255, 255, 255),
    "mode_text": (100, 100, 255),
    "mode_number": (100, 255, 100),
    "mode_pose": (255, 100, 100),
    "success": (0, 255, 0),
    "warning": (0, 165, 255),
    "error": (0, 0, 255),
}

# Detection parameters
DETECTION_COOLDOWN = 1.0
GESTURE_STABILITY_REQUIRED = 5
FACE_PROXIMITY_THRESHOLD = 0.55  # Wrist Y threshold for face proximity
THUMB_INDEX_DISTANCE_THRESHOLD = 0.06

# ============================
# AUDIO INITIALIZATION
# ============================
os.makedirs(TTS_DIR, exist_ok=True)
pygame.mixer.init()
last_spoken = ""
tts_lang = "en"

def speak(text):
    global last_spoken
    if not text or text == last_spoken:
        return
    last_spoken = text

    def worker():
        if not pygame.mixer.get_init():
            return
        filename = os.path.join(TTS_DIR, f"tts_{random.randint(100000, 999999)}.mp3")
        try:
            tts = gTTS(text=text, lang=tts_lang)
            tts.save(filename)
            if pygame.mixer.get_init():
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                while pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                    time.sleep(0.1)
        finally:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except:
                pass

    threading.Thread(target=worker, daemon=True).start()

# ============================
# MEDIAPIPE INITIALIZATION
# ============================
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.85
)
face = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ============================
# CAMERA INITIALIZATION
# ============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("CAMERA NOT DETECTED")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# ============================
# STATE MANAGEMENT
# ============================
custom_gestures = {}
if os.path.exists(CUSTOM_GESTURES_FILE):
    try:
        with open(CUSTOM_GESTURES_FILE, "r") as f:
            custom_gestures = json.load(f)
    except Exception as e:
        print(f"Failed to load custom_gestures.json: {e}")

gesture_counter = {key: 0 for key in GESTURE_TEXTS.keys()}
current_mode = "text"
training_mode = False
recorded_gesture = None
last_detection_time = 0
previous_gesture = ""
gesture_stability_count = 0
subtitle = ""
alpha = 0

# Debug variables
debug_wrist_y = 0.0
debug_is_near_face = False

# ============================
# UTILITY FUNCTIONS
# ============================
def cleanup_tts_files():
    try:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    except:
        pass
    time.sleep(0.2)
    if os.path.exists(TTS_DIR):
        for f in os.listdir(TTS_DIR):
            try:
                os.remove(os.path.join(TTS_DIR, f))
            except:
                pass
        try:
            os.rmdir(TTS_DIR)
        except:
            pass

def adjust_brightness_contrast(frame, brightness=0, contrast=0):
    if brightness != 0:
        alpha_b = 1.0 + brightness / 100.0
        frame = cv2.convertScaleAbs(frame, alpha=alpha_b, beta=0)
    if contrast != 0:
        alpha_c = 1.0 + contrast / 100.0
        frame = cv2.convertScaleAbs(frame, alpha=alpha_c, beta=0)
    return frame

def put_multi_line_text(frame, text, x, y, font_scale, color, thickness):
    lines = text.split('\n')
    y_offset = y
    for line in lines:
        cv2.putText(frame, line, (x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        y_offset += int(40 * font_scale)

def count_fingers(hand):
    tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
    fingers = []
    # Thumb: compare x (horizontal) with landmark 2 (thumb base)
    if hand.landmark[tips[0]].x < hand.landmark[tips[0] - 2].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers: compare y (vertical) with previous landmark
    for i in range(1, 5):
        if hand.landmark[tips[i]].y < hand.landmark[tips[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def log_gesture(gesture):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {gesture}\n")
    except:
        pass

# ============================
# GESTURE DETECTION FUNCTIONS
# ============================
def detect_gesture_text(fingers, hand_lm):
    global debug_wrist_y, debug_is_near_face
    wrist_y = hand_lm.landmark[0].y
    is_near_face = wrist_y < FACE_PROXIMITY_THRESHOLD
    debug_wrist_y = wrist_y
    debug_is_near_face = is_near_face

    # Love you: thumb and index touching
    if fingers == [1, 1, 0, 0, 0]:
        thumb_tip = hand_lm.landmark[4]
        index_tip = hand_lm.landmark[8]
        dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
        if dist < THUMB_INDEX_DISTANCE_THRESHOLD:
            return "love_you"
    
    # Phrases only active when hand is near face
    if is_near_face:
        if fingers == [0, 1, 0, 0, 0]:
            return "my_name"
        if fingers == [0, 1, 1, 0, 0]:
            return "zakiyyah"
        if fingers == [0, 1, 1, 1, 0]:
            return "kiyya"
        if fingers == [0, 0, 0, 0, 0]:
            return "about_me"
    
    # Always active
    if fingers == [1, 1, 1, 1, 1]:
        return "hello"
    if fingers == [1, 1, 0, 0, 1]:
        return "nice_meet"
    
    # Letters
    if fingers == [0, 1, 0, 0, 1]:
        return "a"
    if fingers == [1, 0, 0, 0, 1]:
        return "b"
    if fingers == [1, 0, 0, 0, 0]:
        return "c"
    
    return ""

def detect_gesture_number(fingers, hand_lm):
    if fingers == [0, 0, 0, 0, 0]:
        return "zero"
    if fingers == [0, 1, 0, 0, 0]:
        return "one"
    if fingers == [0, 1, 1, 0, 0]:
        return "two"
    if fingers == [0, 1, 1, 1, 0]:
        return "three"
    if fingers == [0, 1, 1, 1, 1]:
        return "four"
    if fingers == [1, 1, 1, 1, 1]:
        return "five"
    if fingers == [1, 0, 0, 0, 0]:
        return "six"
    if fingers == [1, 1, 0, 0, 0]:
        return "seven"
    if fingers == [1, 1, 1, 0, 0]:
        return "eight"
    if fingers == [0, 1, 0, 0, 1]:
        return "nine"
    if fingers == [1, 1, 1, 1, 0]:
        return "ten"
    return ""

def detect_pose(pose_lm):
    if not pose_lm:
        return ""
    
    l_shoulder = pose_lm.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_shoulder = pose_lm.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    l_wrist = pose_lm.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    r_wrist = pose_lm.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    l_elbow = pose_lm.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    r_elbow = pose_lm.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

    def is_waving(shoulder, wrist, elbow):
        vertical_ok = (shoulder.y - wrist.y) > 0.12
        horizontal_ok = abs(wrist.x - shoulder.x) < 0.35
        elbow_bent = abs(elbow.y - wrist.y) > 0.05
        return vertical_ok and horizontal_ok and elbow_bent

    if is_waving(r_shoulder, r_wrist, r_elbow) or is_waving(l_shoulder, l_wrist, l_elbow):
        return "waving"
    
    return ""

# ============================
# UI SETUP
# ============================
cv2.namedWindow("Hand Gesture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Gesture", 1280, 720)

def nothing(x):
    pass

cv2.createTrackbar("Brightness", "Hand Gesture", 50, 100, nothing)
cv2.createTrackbar("Contrast", "Hand Gesture", 50, 100, nothing)

# ============================
# MAIN LOOP
# ============================
print(" Hand Gesture v2.0 — Gesture Nama Sudah Aktif!")
print("  • Press 't' → Mode TEKS (frasa & nama)")
print("  • Press 'n' → Mode ANGKA (0-10)")
print("  • Press 'p' → Mode POSE (waving)")
print("  • Press 'r' → Toggle Training Mode")
print("  • Press 's' → Save gesture (when Training Mode ON)")
print("  • Press 'l' → Change TTS language (id/en/es)")
print("  • Press 'q' or ESC → Exit")
print("\n Tips: For 'zakiyyah'/'kiyya', raise hand to chest/neck height.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    brightness = cv2.getTrackbarPos("Brightness", "Hand Gesture") - 50
    contrast = cv2.getTrackbarPos("Contrast", "Hand Gesture") - 50
    frame = adjust_brightness_contrast(frame, brightness=brightness, contrast=contrast)
    h, w, _ = frame.shape

    # Face detection for reference
    face_res = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if face_res.detections:
        for det in face_res.detections:
            box = det.location_data.relative_bounding_box
            x1, y1 = int(box.xmin * w), int(box.ymin * h)
            x2, y2 = x1 + int(box.width * w), y1 + int(box.height * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    gesture_text = ""
    pose_text = ""
    fingers = [0, 0, 0, 0, 0]

    # Hand detection
    hand_res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if hand_res.multi_hand_landmarks:
        for hand_lm in hand_res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            fingers = count_fingers(hand_lm)
            
            if current_mode == "text":
                gesture_text = detect_gesture_text(fingers, hand_lm)
            elif current_mode == "number":
                gesture_text = detect_gesture_number(fingers, hand_lm)

    # Pose detection
    pose_res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if pose_res.pose_landmarks and current_mode == "pose":
        mp_draw.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        pose_text = detect_pose(pose_res.pose_landmarks)

    # Gesture smoothing
    if gesture_text == previous_gesture:
        gesture_stability_count += 1
    else:
        gesture_stability_count = 0
        previous_gesture = gesture_text

    stable_gesture = gesture_text if gesture_stability_count >= GESTURE_STABILITY_REQUIRED else ""
    combined_text = stable_gesture or pose_text

    # Cooldown and action
    current_time = time.time()
    if combined_text and (current_time - last_detection_time) > DETECTION_COOLDOWN:
        if training_mode:
            recorded_gesture = combined_text
            subtitle = f"Recorded: '{combined_text}' | Press 's' to save"
        else:
            subtitle = GESTURE_TEXTS.get(combined_text, combined_text)
            gesture_counter[combined_text] = gesture_counter.get(combined_text, 0) + 1
            log_gesture(combined_text)
            alpha = 255
            speak(subtitle)
        last_detection_time = current_time

    # Draw UI
    # Subtitle fade
    if alpha > 0:
        overlay = frame.copy()
        put_multi_line_text(overlay, subtitle, 50, h - 130, 1.5, COLORS["subtitle"], 2)
        frame = cv2.addWeighted(overlay, alpha / 255, frame, 1 - alpha / 255, 0)
        alpha = max(0, alpha - 6)

    # Mode indicator
    mode_color = COLORS["mode_text"]
    if current_mode == "number":
        mode_color = COLORS["mode_number"]
    elif current_mode == "pose":
        mode_color = COLORS["mode_pose"]
    cv2.putText(frame, f"MODE: {current_mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

    # Debug info
    cv2.putText(frame, f"Fingers: {fingers}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["debug"], 1)
    wrist_status = "✅ Near Face" if debug_is_near_face else " Too Low"
    wrist_color = COLORS["success"] if debug_is_near_face else COLORS["warning"]
    cv2.putText(frame, f"Wrist Y: {debug_wrist_y:.2f} | {wrist_status}",
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, wrist_color, 1)
    
    cv2.putText(frame, f"Gesture: '{gesture_text}'", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["debug"], 1)
    cv2.putText(frame, f"Stable: {'YES' if stable_gesture else 'NO'} ({gesture_stability_count}/{GESTURE_STABILITY_REQUIRED})",
                (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["debug"], 1)

    # FPS and author
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["fps"], 2)
    cv2.putText(frame, AUTHOR_NAME, (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["author"], 2)

    # Counter (last 3 active gestures)
    active = [f"{k}:{v}" for k, v in gesture_counter.items() if v > 0][-3:]
    if active:
        cv2.putText(frame, " | ".join(active), (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["counter"], 1)

    # Training indicator
    if training_mode:
        cv2.putText(frame, "TRAINING MODE", (w//2 - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS["error"], 2)

    cv2.imshow("Hand Gesture", frame)

    # Key handler
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('t'):
        current_mode = "text"
        subtitle = "Mode: TEKS (frasa & nama)"
        alpha = 255
    elif key == ord('n'):
        current_mode = "number"
        subtitle = "#️Mode: ANGKA (0-10)"
        alpha = 255
    elif key == ord('p'):
        current_mode = "pose"
        subtitle = "Mode: POSE (waving)"
        alpha = 255
    elif key == ord('r'):
        training_mode = not training_mode
        subtitle = " TRAINING MODE ON" if training_mode else " TRAINING MODE OFF"
        alpha = 255
    elif key == ord('s') and training_mode and recorded_gesture:
        try:
            print(f"\n Save gesture '{recorded_gesture}' as...")
            custom_text = input("Text: ").strip()
            if custom_text:
                custom_gestures[recorded_gesture] = {"fingers": fingers, "text": custom_text}
                with open(CUSTOM_GESTURES_FILE, "w") as f_out:
                    json.dump(custom_gestures, f_out, indent=2, ensure_ascii=False)
                GESTURE_TEXTS[recorded_gesture] = custom_text
                subtitle = f" Saved: '{custom_text}'"
                alpha = 255
            else:
                subtitle = "❌ Cancelled: empty text"
                alpha = 255
        except Exception as e:
            subtitle = f"❌ Error: {e}"
            alpha = 255
        recorded_gesture = None
    elif key == ord('l'):
        print("\n TTS Language: id (Indonesia), en (English), es (Spanish), fr (French)")
        new_lang = input("Language code: ").strip().lower()
        if new_lang in ["id", "en", "es", "fr", "ja", "ko", "zh"]:
            tts_lang = new_lang
            lang_names = {"id": "Indonesia", "en": "English", "es": "Spanish", "fr": "French", "ja": "Japanese", "ko": "Korean", "zh": "Mandarin"}
            subtitle = f"🔊 Language: {lang_names.get(new_lang, new_lang)}"
            alpha = 255
        else:
            subtitle = "❌ Unsupported code"
            alpha = 255

    # Check if window is closed
    if cv2.getWindowProperty("Hand Gesture", cv2.WND_PROP_VISIBLE) < 1:
        break

# Cleanup
cleanup_tts_files()
cap.release()
cv2.destroyAllWindows()
print("\n Program finished. Thank you for using Hand Gesture!")