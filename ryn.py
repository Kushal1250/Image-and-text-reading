import os
import cv2
import re
import pytesseract
import threading
import webbrowser
import numpy as np
import pyttsx3
import playsound
import speech_recognition as sr
from datetime import datetime
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "yolov8x.pt"
SNAPSHOT_DIR = "snapshots"
LOG_FILE = "detections_log.txt"
DING_FILE = "ding.mp3"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# === LOAD MODEL ===
if not os.path.exists(MODEL_PATH):
    print("📦 Downloading YOLOv8x model...")
model = YOLO(MODEL_PATH)

# === INIT OCR ===
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# === INIT CAMERA ===
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# === INIT SPEECH ===
engine = pyttsx3.init()
engine.setProperty('rate', 170)

# === GLOBALS ===
paused = False
show_details = False
main_info = ""
last_detected = ""

print("\n🔍 Show object/text to the camera")
print("🎛️ Press P-pause | D-details | R-read | V-voice | Q-quit\n")

# === UTILS ===
def speak(text):
    print(f"🔈 Speaking: {text}")
    engine.say(text)
    engine.runAndWait()

def preprocess_for_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.equalizeHist(gray)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 21, 10)

def clean_ocr(text):
    return re.sub(r'[^\x20-\x7E\n]', '', text).strip()

def draw_text(img, text, pos, color=(255,255,255), bg=(0,0,0), scale=0.6, thick=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate(text.split('\n')):
        (w, h), _ = cv2.getTextSize(line, font, scale, thick)
        y = pos[1] + i * (h + 10)
        cv2.rectangle(img, (pos[0], y - h - 10), (pos[0] + w + 10, y + 10), bg, -1)
        cv2.putText(img, line, (pos[0] + 5, y), font, scale, color, thick)

def draw_box_label(img, label, box, color=(0,255,0)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    draw_text(img, label, (x1, y1 - 5), color=color, bg=(0, 0, 0))

def google(query):
    # Open the search page
    webbrowser.open(f"https://www.google.com/search?q={query}")

    # Append to google_result.txt
    with open("google_result.txt", "a", encoding="utf-8") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"[{timestamp}] {query}\n")
        print(f"📝 Logged Google search: {query}")


def log_and_snapshot(objects, frame):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {objects}\n")
    cv2.imwrite(f"{SNAPSHOT_DIR}/detected_{timestamp}.jpg", frame)

def play_ding():
    if os.path.exists(DING_FILE):
        threading.Thread(target=playsound.playsound, args=(DING_FILE,), daemon=True).start()

def voice_command():
    try:
        recog = sr.Recognizer()
        with sr.Microphone() as source:
            print("🎙️ Listening...")
            recog.adjust_for_ambient_noise(source)
            audio = recog.listen(source)
            command = recog.recognize_google(audio).lower()
            print("🎤 You said:", command)
            return command
    except:
        print("❌ Voice input failed.")
        return ""

def detail_menu(query):
    print(f"\n🔍 Options for: '{query}'\n1. Search on Google")
    if input("Enter option (1): ") == "1":
        google(query)

# === MAIN LOOP ===
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera error.")
            break

        annotated = frame.copy()
        results = model.predict(frame, conf=0.35, verbose=False)[0]

        labels = []
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            name = model.model.names[int(cls)]
            label_text = f"{name} ({float(conf):.2f})"
            draw_box_label(annotated, label_text, box)
            labels.append(name)

        if labels:
            detected = ', '.join(set(labels))
            draw_text(annotated, f"Object(s): {detected}", (10, 30), (0, 255, 0))
            if detected != last_detected:
                main_info = detected
                last_detected = detected
                play_ding()
                log_and_snapshot(detected, annotated)

        # OCR
        ocr_img = preprocess_for_ocr(frame)
        raw_text = pytesseract.image_to_string(ocr_img, lang='eng')
        lines = [line for line in clean_ocr(raw_text).split('\n') if len(line.strip()) > 3]
        if lines:
            longest = max(lines, key=len)
            draw_text(annotated, f"Text: {longest}", (10, 70), (0, 255, 255))
            if not main_info:
                main_info = longest
            if show_details:
                draw_text(annotated, "\n".join(lines[:5]), (10, 110), (255, 200, 0))

        # Timestamp
        draw_text(annotated, f"Time: {datetime.now().strftime('%H:%M:%S')}",
                  (10, annotated.shape[0] - 20), (255,255,255), (50,50,50))

        cv2.imshow("📷 Smart Vision", annotated)
    else:
        cv2.imshow("📷 Smart Vision", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("❌ Exiting...")
        break
    elif key == ord('p'):
        paused = not paused
        print("⏸️ Paused" if paused else "▶️ Resumed")
    elif key == ord('r') and main_info:
        threading.Thread(target=speak, args=(main_info,), daemon=True).start()
    elif key == ord('d') and main_info:
        threading.Thread(target=detail_menu, args=(main_info,), daemon=True).start()
        show_details = not show_details
    elif key == ord('v'):
        command = voice_command()
        if "pause" in command:
            paused = True
        elif "resume" in command:
            paused = False
        elif "quit" in command:
            break
        elif "details" in command and main_info:
            threading.Thread(target=detail_menu, args=(main_info,), daemon=True).start()
            show_details = not show_details

cap.release()
cv2.destroyAllWindows()
