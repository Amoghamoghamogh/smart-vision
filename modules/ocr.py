# modules/ocr.py — PaddleOCR text detection and recognition pipeline.
#
# Pipeline (mirrors the presentation's Figure 3):
#   Frame → DBNet text detection → direction classifier → CRNN recognition
#   → text string → TTS queue
#
# Frame skipping is already handled upstream by CameraThread.
# Here we apply an additional confidence filter (OCR_CONF_THRESH) to avoid
# announcing low-quality reads.
#
# The redundancy check (SIMILARITY_THRESHOLD) prevents the TTS thread from
# repeating the same text when the camera is stationary.

import logging
import threading
from queue import Queue, Empty

from config import OCR_LANG, OCR_USE_GPU, OCR_CONF_THRESH, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


def _similarity(a: str, b: str) -> float:
    """
    Simple character-level Jaccard similarity between two strings.
    Fast enough for short OCR outputs; no external dependency needed.
    """
    if not a or not b:
        return 0.0
    set_a, set_b = set(a.lower()), set(b.lower())
    return len(set_a & set_b) / len(set_a | set_b)


def _load_ocr():
    """Load PaddleOCR. Returns None if paddleocr is not installed."""
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang=OCR_LANG,
                        use_gpu=OCR_USE_GPU, show_log=False)
        logger.info("PaddleOCR loaded (lang=%s, gpu=%s)", OCR_LANG, OCR_USE_GPU)
        return ocr
    except Exception as e:
        logger.error("Failed to load PaddleOCR: %s", e)
        return None


class OCRThread(threading.Thread):
    """
    Pulls frames from frame_queue, runs PaddleOCR, and pushes recognised
    text to tts_queue.

    Only runs when mode_controller.mode == "ocr".
    Applies similarity check to suppress repeated announcements.
    """

    def __init__(self, frame_queue: Queue, tts_queue: Queue,
                 mode_controller, stop_event: threading.Event):
        super().__init__(name="OCRThread", daemon=True)
        self.frame_queue     = frame_queue
        self.tts_queue       = tts_queue
        self.mode_controller = mode_controller
        self.stop_event      = stop_event
        self._ocr            = None
        self._last_text      = ""

    def run(self):
        self._ocr = _load_ocr()

        while not self.stop_event.is_set():
            if self.mode_controller.mode != "ocr":
                self.stop_event.wait(timeout=0.1)
                continue

            try:
                frame = self.frame_queue.get(timeout=0.5)
            except Empty:
                continue

            text = self._run_ocr(frame)
            if not text:
                continue

            # Redundancy check: skip if >80% similar to last spoken text
            if _similarity(text, self._last_text) >= SIMILARITY_THRESHOLD:
                logger.debug("OCR: suppressed repeated text")
                continue

            self._last_text = text
            logger.debug("OCR recognised: %s", text)
            self.tts_queue.put(text)

    def _run_ocr(self, frame) -> str:
        """Run PaddleOCR on a frame and return concatenated text above threshold."""
        if self._ocr is None:
            return "[stub] OCR not available"

        try:
            results = self._ocr.ocr(frame, cls=True)
        except Exception as e:
            logger.error("OCR inference error: %s", e)
            return ""

        if not results or not results[0]:
            return ""

        lines = []
        for line in results[0]:
            text, confidence = line[1]
            if confidence >= OCR_CONF_THRESH:
                lines.append(text)

        return " ".join(lines).strip()
