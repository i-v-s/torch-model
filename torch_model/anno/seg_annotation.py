from typing import Optional

import cv2
import numpy as np

from torch_model import process_image
from .annotation import Annotation


class SegAnnotation(Annotation):
    """Annotation for segmentation using bitmap mask"""
    def __init__(self, channels=3, model_name: Optional[str] = None, device=None, radius=10, opacity: float = 0.7):
        super().__init__(model_name, device)
        if type(channels) is int:
            channels = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255))[:channels]
        self.channels = np.array(channels, dtype=np.uint8)
        self.opacity = opacity
        self.active_channel = 0
        self.mask = None
        self.keys = {ord('1') + c: c for c in range(len(channels))}
        self.start_pos = None
        self.radius = radius

    def on_key(self, key):
        c = self.keys.get(key, None)
        if c is not None:
            self.active_channel = c
        elif key == ord('='):
            self.radius += 1
        elif key == ord('-'):
            if self.radius > 1:
                self.radius -= 1
        else:
            return False
        return True

    def circle(self, x, y, col):
        mask = self.mask[:, :, self.active_channel]
        mm = mask.copy()
        cv2.circle(mm, (x, y), self.radius, (col,), -1)
        mask[:] = mm

    def line(self, x, y, col):
        mask = self.mask[:, :, self.active_channel]
        mm = mask.copy()
        cv2.line(mm, self.start_pos, (x, y), (col,), thickness=self.radius * 2)
        mask[:] = mm

    def on_left_down(self, x, y):
        self.circle(x, y, 255)
        self.start_pos = x, y

    def on_left_up(self, x: int, y: int):
        self.circle(x, y, 255)
        self.line(x, y, 255)

    def on_right_down(self, x: int, y: int):
        self.circle(x, y, 0)
        self.start_pos = x, y

    def on_right_up(self, x: int, y: int):
        self.circle(x, y, 0)
        self.line(x, y, 0)

    def visualize(self, image: np.ndarray, cursor, frame_no):
        if self.mask is None:
            shape = image.shape[:2]
            self.mask = np.zeros(shape + (len(self.channels),), dtype=image.dtype)
        mask = (np.expand_dims(self.mask, 3).astype(np.float32) * self.channels).sum(2) * (self.opacity / 255)
        image[:] = np.clip(image + mask, 0, 255).astype(np.uint8)
        if cursor is not None:
            x, y = cursor
            cv2.circle(image, (x, y), self.radius, (255, 255, 255), -1)

    def process(self, frame: np.ndarray):
        self.mask = process_image(frame, self.model)

    def save(self, fn: str):
        if len(self.channels) > 3:
            raise NotImplementedError
        mask = self.mask
        # if len(mask.shape) == 3 and mask.shape[2] in ['2']:
        cv2.imwrite(fn + '.png', mask)
        return True

    def clear(self):
        self.mask[:] = 0

    def load(self, fn: str):
        self.mask = cv2.imread(fn + '.png')