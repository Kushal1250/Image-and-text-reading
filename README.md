# 🧠🔍 Image and Text Reading System

A smart vision-based AI project that performs **real-time object detection** using YOLOv8 and **text extraction** from images and live camera feeds using OCR (Optical Character Recognition). This solution is useful for automated surveillance, document scanning, accessibility tools, and smart assistants.

![Python](https://img.shields.io/badge/Built%20With-Python-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/Object%20Detection-YOLOv8-orange?style=flat-square)
![OCR](https://img.shields.io/badge/OCR-Pytesseract-yellow?style=flat-square)

---

## 📌 Features

- 🎯 Real-time Object Detection (YOLOv8)
- 🧾 Text Recognition from Images (OCR)
- 📸 Automatic Snapshot on Detection
- 🔊 Audio Alert on Detection (`ding.mp3`)
- 🧠 ChatGPT Integration for Smart Summary
- 📂 Snapshot & Log Storage
- 📜 Instruction and Result Logs
- 🚀 Easy-to-run Script

---

## 📁 Project Structure

Image-and-text-reading/
├── snapshots/ # Auto-saved images with detections
├── yolov8n.pt # Lightweight YOLOv8 model (not pushed due to size limits)
├── yolov8x.pt # High-accuracy model (Download externally)
├── ryn.py # Main script
├── intructruction.txt # Usage guide and notes
├── detections_log.txt # Log of all detections
├── google_result.txt # Captured OCR results
├── chatgpt_result.html # Response from ChatGPT integration
├── ding.mp3.mp3 # Notification sound
└── README.md # Project overview


---

## 🧠 Requirements

- Python 3.8+
- OpenCV
- Ultralytics
- pytesseract
- playsound
- pyttsx3
- torch
- transformers (if using ChatGPT API)

> Install all dependencies with:
```bash
pip install -r requirements.txt


Step 1: Download YOLOv8 Model
Due to GitHub’s 100MB limit, download yolov8x.pt manually:

YOLOv8x.pt (from Ultralytics)

Place it in the project root folder.

Step 2: Run the Script
python ryn.py
