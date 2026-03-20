# modules/detection.py — YOLOv8 object detection inference.
#
# Loads a YOLOv8 model once at startup and runs inference on frames pulled
# from the camera queue. Detected class names above the confidence threshold
# are pushed to the TTS queue as a comma-separated announcement string.
#
# Model choice: yolov8n (nano) — best speed/accuracy tradeoff on CM5 CPU.
# Inference image size is reduced to 320px to further cut latency.

import logging
import threading
from queue import Queue, Empty

from config import YOLO_MODEL_PATH, YOLO_CONF_THRESH, YOLO_IMG_SIZE

logger = logging.getLogger(__name__)


def _load_model():
    """Load YOLOv8 model. Returns None if ultralytics is not installed."""
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL_PATH)
        logger.info("YOLOv8 model loaded: %s", YOLO_MODEL_PATH)
        return model
    except Exception as e:
        logger.error("Failed to load YOLO model: %s", e)
        return None


class DetectionThread(threading.Thread):
    """
    Pulls frames from frame_queue, runs YOLOv8 inference, and pushes
    detected object labels to tts_queue.

    Only runs inference when mode_controller.mode == "detection".
    """

    def __init__(self, frame_queue: Queue, tts_queue: Queue,
                 mode_controller, stop_event: threading.Event):
        super().__init__(name="DetectionThread", daemon=True)
        self.frame_queue    = frame_queue
        self.tts_queue      = tts_queue
        self.mode_controller = mode_controller
        self.stop_event     = stop_event
        self._model         = None

    def run(self):
        self._model = _load_model()

        while not self.stop_event.is_set():
            if self.mode_controller.mode != "detection":
                # Not our turn — yield CPU
                self.stop_event.wait(timeout=0.1)
                continue

            try:
                frame = self.frame_queue.get(timeout=0.5)
            except Empty:
                continue

            if self._model is None:
                # Stub output for environments without ultralytics installed
                self.tts_queue.put("[stub] object detection not available")
                continue

            results = self._model.predict(
                source=frame,
                conf=YOLO_CONF_THRESH,
                imgsz=YOLO_IMG_SIZE,
                verbose=False,
            )

            labels = self._extract_labels(results)
            if labels:
                announcement = "Detected: " + ", ".join(labels)
                logger.debug(announcement)
                self.tts_queue.put(announcement)

    @staticmethod
    def _extract_labels(results) -> list[str]:
        """Return unique class names from YOLO results above conf threshold."""
        seen = set()
        for result in results:
            for box in result.boxes:
                name = result.names[int(box.cls)]
                if name not in seen:
                    seen.add(name)
        return list(seen)
