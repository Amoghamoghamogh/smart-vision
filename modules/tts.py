# modules/tts.py — offline Text-to-Speech output thread.
#
# Consumes text strings from tts_queue and vocalises them using pyttsx3
# (offline, no network required — suitable for rural/low-connectivity use).
#
# pyttsx3 uses eSpeak on Linux (available on Raspberry Pi OS).
# For higher quality speech, swap the _speak() implementation with a
# VITTS / Glow-TTS inference call — the queue interface stays the same.
#
# Queue design:
#   - Bounded queue (TTS_QUEUE_MAXSIZE) prevents unbounded backlog.
#   - If the queue is full, the oldest item is dropped (camera/inference
#     threads use put_nowait and catch Full).

import logging
import threading
from queue import Queue, Empty

from config import TTS_RATE, TTS_VOLUME, TTS_QUEUE_MAXSIZE

logger = logging.getLogger(__name__)


def _load_engine():
    """Load pyttsx3 TTS engine. Returns None if not installed."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate",   TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)
        logger.info("pyttsx3 TTS engine initialised")
        return engine
    except Exception as e:
        logger.error("Failed to initialise TTS engine: %s", e)
        return None


class TTSThread(threading.Thread):
    """
    Dedicated thread for audio output.
    Runs independently so inference threads are never blocked by speech duration.
    """

    def __init__(self, tts_queue: Queue, stop_event: threading.Event):
        super().__init__(name="TTSThread", daemon=True)
        self.tts_queue  = tts_queue
        self.stop_event = stop_event
        self._engine    = None

    def run(self):
        self._engine = _load_engine()

        while not self.stop_event.is_set():
            try:
                text = self.tts_queue.get(timeout=0.5)
            except Empty:
                continue

            self._speak(text)

    def _speak(self, text: str):
        if not text:
            return

        if self._engine is None:
            # Fallback: print to stdout (useful during development)
            print(f"[TTS] {text}")
            return

        try:
            logger.debug("Speaking: %s", text)
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            logger.error("TTS error: %s", e)

    def announce(self, text: str):
        """
        Convenience method to push a message directly (e.g. welcome message).
        Bypasses the queue for immediate playback.
        """
        self._speak(text)
