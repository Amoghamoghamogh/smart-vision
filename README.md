# Smart Vision for Visually Impaired

Real-time assistive system on Raspberry Pi CM5 — detects objects and reads text aloud using fully offline, on-device AI.

---

## Overview

Visually impaired users have limited access to affordable, offline assistive tools. This system embeds a camera into a glasses frame and runs a multithreaded AI pipeline directly on a Raspberry Pi CM5. Three physical buttons switch between modes. Everything — inference, speech, control — runs without internet or a screen.

<!-- SUGGESTED DIAGRAM: place a pipeline flow diagram here (assets/architecture.png)
     showing Camera → frame_queue → Detection/OCR threads → tts_queue → TTS → Speaker -->

---

## Architecture

```
Camera (OpenCV)
      │
      ▼  1 in 10 frames (frame skipping)
  frame_queue  maxsize=2  ← stale frames dropped, not queued
      │                    │
      ▼                    ▼
DetectionThread        OCRThread
(YOLOv8n)             (PaddleOCR DBNet + CRNN)
      │                    │
      └────────┬───────────┘
               ▼
          tts_queue  maxsize=5  ← oldest dropped if full
               │
               ▼
          TTSThread (pyttsx3 / VITTS)
               │
               ▼
          Speaker output
```

Threading model:
- 4 daemon threads: `CameraThread`, `DetectionThread`, `OCRThread`, `TTSThread`
- All inter-thread communication via bounded `Queue` — no shared mutable state
- `ModeController` holds the active mode behind a `threading.Lock`; GPIO callbacks use 300ms debounce
- Only the active inference thread processes frames — the other idles with a 100ms sleep

---

## Key Design Decisions

**frame_queue maxsize=2, non-blocking puts**
Camera captures at ~30fps; inference runs at ~3–5fps on CM5 CPU. A bounded queue with `put_nowait` + `Full` discard ensures inference always sees the latest frame, not a frame from 2 seconds ago. An unbounded queue would cause latency to grow indefinitely.

**tts_queue maxsize=5**
Speech takes 1.5–2.5s per announcement. If inference produces results faster than TTS can consume them, audio becomes temporally irrelevant ("chair" announced 4 seconds after the user has moved on). Capping at 5 and dropping the oldest keeps speech current.

**No blocking queue operations in inference threads**
Both `DetectionThread` and `OCRThread` use `queue.get(timeout=0.5)` — they never stall waiting for a frame. This keeps the mode-switch response time under 1.5s regardless of inference state.

**OCR redundancy filter (Jaccard similarity)**
When the camera is stationary, PaddleOCR produces near-identical results every 10 frames. A character-level Jaccard similarity check suppresses output above 80% similarity to the last spoken string — prevents the device from repeating itself.

**Swappable TTS**
`_speak()` in `tts.py` is the only method that changes when upgrading from pyttsx3/eSpeak to VITTS or Glow-TTS. Queue interface and thread lifecycle are untouched.

**Graceful hardware fallback**
`RPi.GPIO`, `ultralytics`, and `paddleocr` are all wrapped in try/except. The pipeline runs on any machine — inference threads emit stub output, GPIO uses a software `set_mode()` override. Useful for development and CI.

---

## Features

- YOLOv8n object detection — optimised for edge CPU (320px inference, 0.50 conf threshold)
- PaddleOCR pipeline: DBNet text detection → direction classifier → CRNN recognition
- Offline TTS via pyttsx3/eSpeak — zero network dependency
- GPIO mode switching: object detection / OCR / shutdown
- Frame skipping (1 in 10) to prevent inference queue backlog
- Redundancy filter suppresses repeated OCR announcements
- Auto-starts on boot via systemd service
- Runs without Raspberry Pi hardware (stub fallbacks for all hardware dependencies)

---

## Project Structure

```
smart-vision/
├── main.py                  # thread orchestration, boot sequence, shutdown
├── config.py                # all tunable constants (GPIO pins, thresholds, queue sizes)
├── requirements.txt
└── modules/
    ├── camera.py            # CameraThread — capture + frame skipping
    ├── detection.py         # DetectionThread — YOLOv8 inference
    ├── ocr.py               # OCRThread — PaddleOCR + Jaccard redundancy filter
    ├── tts.py               # TTSThread — offline speech, swappable backend
    └── gpio_control.py      # ModeController — GPIO interrupts, lock, debounce
```

---

## Setup & Run

Requirements: Python 3.10+, eSpeak (Linux), optionally RPi.GPIO on Raspberry Pi.

```bash
git clone https://github.com/<your-username>/smart-vision.git
cd smart-vision

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# eSpeak for offline TTS (Raspberry Pi OS / Debian)
sudo apt install espeak

python main.py
```

On boot the system announces: *"Smart Vision system ready. Press a button to begin."*

To auto-start on boot:

```bash
sudo cp deploy/smart-vision.service /etc/systemd/system/
sudo systemctl enable --now smart-vision
```

GPIO 3 → object detection mode | GPIO 4 → OCR mode | GPIO 2 → shutdown

<!-- SUGGESTED SCREENSHOT: place a terminal screenshot here (assets/demo-output.png)
     showing live detection output and TTS announcements in the console log -->

---

## Performance

Measured on Raspberry Pi CM5 (4GB RAM, no GPU):

| Metric | Value |
|--------|-------|
| Mode switch latency | 1.2 – 1.5 s |
| Audio output after detection | 1.5 – 2.5 s |
| Frame skip rate | 1 in 10 |
| YOLO inference resolution | 320 × 320 px |
| OCR confidence threshold | 0.70 |
| Redundancy suppression threshold | 0.80 (Jaccard) |

---

## Limitations

- OCR accuracy degrades under poor lighting or motion blur
- YOLOv8n is COCO-trained — uncommon or domain-specific objects may not be detected
- pyttsx3/eSpeak produces robotic speech compared to neural TTS
- No distance estimation — system cannot tell how far away a detected object is
- Single OCR language per session (configured in `config.py`)

---

## Future Work

- [ ] Replace eSpeak with VITTS / Glow-TTS for natural-sounding speech
- [ ] Add depth sensor (OAK-D) for distance-to-object estimation
- [ ] Multilingual OCR support (Kannada, Hindi, etc.)
- [ ] Wake-word voice commands for hands-free mode switching
- [ ] Scene description using a lightweight VLM (e.g. MobileVLM)
- [ ] INT8 quantization to reduce YOLO inference latency by ~2×

---

## Acknowledgements

University Guide: Prof. Sheela A. B., KLE Tech, Hubballi
Industry Guide: Mr. Manoj Bhat, CEO, Einetcorp Pvt. Ltd.
