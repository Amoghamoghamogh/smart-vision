# config.py — all tunable constants in one place.
# Change values here without touching pipeline logic.

# ── GPIO pin assignments ────────────────────────────────────────────────────
GPIO_POWER  = 2   # power / shutdown button
GPIO_DETECT = 3   # activate object detection mode
GPIO_OCR    = 4   # activate OCR / reading mode

# ── Camera ──────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0      # /dev/video0 on CM5
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
FRAME_SKIP     = 10     # process every Nth frame to reduce CPU load

# ── Object detection ────────────────────────────────────────────────────────
YOLO_MODEL_PATH   = "models/yolov8n.pt"   # nano variant for edge speed
YOLO_CONF_THRESH  = 0.50                  # minimum confidence to announce
YOLO_IMG_SIZE     = 320                   # smaller = faster on CM5

# ── OCR ─────────────────────────────────────────────────────────────────────
OCR_LANG          = "en"    # PaddleOCR language code
OCR_USE_GPU       = False   # CM5 has no discrete GPU
OCR_CONF_THRESH   = 0.70    # ignore low-confidence text regions

# ── TTS ─────────────────────────────────────────────────────────────────────
TTS_RATE          = 150     # words per minute (pyttsx3 fallback)
TTS_VOLUME        = 1.0
TTS_QUEUE_MAXSIZE = 5       # drop oldest if queue fills (prevents lag)

# ── Redundancy filter ───────────────────────────────────────────────────────
# Skip TTS output if new text is >80% similar to the last spoken text.
SIMILARITY_THRESHOLD = 0.80
