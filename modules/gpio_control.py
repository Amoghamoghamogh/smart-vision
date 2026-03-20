# modules/gpio_control.py — GPIO button handling and operating mode state.
#
# Three physical buttons on GPIO 2/3/4 control the system mode.
# Mode is stored as a thread-safe string so all threads can read it.
#
# Modes:
#   "idle"      — system on, no inference running
#   "detection" — YOLOv8 object detection active
#   "ocr"       — PaddleOCR reading mode active

import threading
import logging

logger = logging.getLogger(__name__)

# Try to import RPi.GPIO; fall back to a stub so the code runs on non-Pi hardware.
try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:
    logger.warning("RPi.GPIO not found — running in stub mode (no physical buttons)")
    _GPIO_AVAILABLE = False

from config import GPIO_POWER, GPIO_DETECT, GPIO_OCR


class ModeController:
    """
    Manages the current operating mode and GPIO interrupt callbacks.
    Thread-safe: mode is protected by a Lock.
    """

    def __init__(self):
        self._mode = "idle"
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

    # ── Mode access ──────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @mode.setter
    def mode(self, value: str):
        with self._lock:
            logger.info("Mode changed: %s → %s", self._mode, value)
            self._mode = value

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_event.is_set()

    # ── GPIO setup ───────────────────────────────────────────────────────────

    def setup(self):
        """Configure GPIO pins with pull-up resistors and attach callbacks."""
        if not _GPIO_AVAILABLE:
            logger.info("GPIO stub active — use set_mode() to switch modes manually")
            return

        GPIO.setmode(GPIO.BCM)
        for pin in (GPIO_POWER, GPIO_DETECT, GPIO_OCR):
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        GPIO.add_event_detect(GPIO_POWER,  GPIO.FALLING,
                              callback=self._on_power,  bouncetime=300)
        GPIO.add_event_detect(GPIO_DETECT, GPIO.FALLING,
                              callback=self._on_detect, bouncetime=300)
        GPIO.add_event_detect(GPIO_OCR,    GPIO.FALLING,
                              callback=self._on_ocr,    bouncetime=300)

        logger.info("GPIO configured on pins %d/%d/%d",
                    GPIO_POWER, GPIO_DETECT, GPIO_OCR)

    def cleanup(self):
        if _GPIO_AVAILABLE:
            GPIO.cleanup()

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_power(self, channel):
        logger.info("Power button pressed — requesting shutdown")
        self._shutdown_event.set()

    def _on_detect(self, channel):
        self.mode = "detection"

    def _on_ocr(self, channel):
        self.mode = "ocr"

    # ── Manual override (testing / non-Pi environments) ──────────────────────

    def set_mode(self, mode: str):
        """Programmatically set mode — useful for testing without hardware."""
        self.mode = mode
