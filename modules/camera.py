# modules/camera.py — camera capture thread.
#
# Reads frames from the camera in a dedicated thread and puts every Nth frame
# into a shared queue for downstream inference modules.
# Frame skipping (FRAME_SKIP) prevents the inference queue from filling faster
# than it can be consumed, which would cause unbounded latency on CM5.

import cv2
import threading
import logging
from queue import Queue, Full

from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FRAME_SKIP

logger = logging.getLogger(__name__)


class CameraThread(threading.Thread):
    """
    Continuously captures frames and forwards sampled frames to a queue.

    Args:
        frame_queue: shared Queue consumed by detection/OCR threads.
        stop_event:  threading.Event — set this to stop the thread cleanly.
    """

    def __init__(self, frame_queue: Queue, stop_event: threading.Event):
        super().__init__(name="CameraThread", daemon=True)
        self.frame_queue = frame_queue
        self.stop_event  = stop_event
        self._cap        = None

    def run(self):
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not self._cap.isOpened():
            logger.error("Failed to open camera at index %d", CAMERA_INDEX)
            return

        logger.info("Camera started (%dx%d, skip=%d)",
                    FRAME_WIDTH, FRAME_HEIGHT, FRAME_SKIP)

        frame_count = 0
        while not self.stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Camera read failed — retrying")
                continue

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue  # drop this frame

            try:
                # Non-blocking put: discard frame if queue is full
                # (inference is slower than capture — this is expected)
                self.frame_queue.put_nowait(frame)
            except Full:
                pass  # inference thread hasn't consumed the previous frame yet

        self._cap.release()
        logger.info("Camera stopped")
