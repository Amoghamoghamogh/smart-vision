# main.py — Smart Vision pipeline entry point.
#
# Thread layout:
#
#   CameraThread  ──► frame_queue ──► DetectionThread ──► tts_queue ──► TTSThread
#                                └──► OCRThread       ──►
#
# ModeController (GPIO) determines which inference thread is active.
# Only one inference thread processes frames at a time — the other idles.
#
# Boot sequence (mirrors the presentation's systemd auto-start):
#   1. Initialise GPIO
#   2. Play welcome message
#   3. Start all threads
#   4. Wait for shutdown button (GPIO 2) or KeyboardInterrupt

import logging
import threading
from queue import Queue

from config import TTS_QUEUE_MAXSIZE
from modules.gpio_control import ModeController
from modules.camera       import CameraThread
from modules.detection    import DetectionThread
from modules.ocr          import OCRThread
from modules.tts          import TTSThread

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # ── Shared state ─────────────────────────────────────────────────────────
    stop_event   = threading.Event()
    frame_queue  = Queue(maxsize=2)          # small — we want latest frame only
    tts_queue    = Queue(maxsize=TTS_QUEUE_MAXSIZE)
    mode_ctrl    = ModeController()

    # ── GPIO setup ───────────────────────────────────────────────────────────
    mode_ctrl.setup()

    # ── Build threads ────────────────────────────────────────────────────────
    tts_thread       = TTSThread(tts_queue, stop_event)
    camera_thread    = CameraThread(frame_queue, stop_event)
    detection_thread = DetectionThread(frame_queue, tts_queue, mode_ctrl, stop_event)
    ocr_thread       = OCRThread(frame_queue, tts_queue, mode_ctrl, stop_event)

    threads = [tts_thread, camera_thread, detection_thread, ocr_thread]

    # ── Start ────────────────────────────────────────────────────────────────
    logger.info("Starting Smart Vision system")
    for t in threads:
        t.start()

    # Welcome message — played immediately via TTSThread
    tts_thread.announce("Smart Vision system ready. Press a button to begin.")

    # ── Wait for shutdown ────────────────────────────────────────────────────
    try:
        while not mode_ctrl.shutdown_requested:
            stop_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down")

    # ── Graceful shutdown ────────────────────────────────────────────────────
    logger.info("Stopping all threads")
    stop_event.set()

    for t in threads:
        t.join(timeout=3.0)

    mode_ctrl.cleanup()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
