import numpy as np
from typing import Tuple


class AnnoTransform:
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        ...

    def invert_point(self, x: int, y: int) -> Tuple[float, float]:
        ...


class CropTransform(AnnoTransform):
    def __init__(self, sx: slice, sy: slice):
        super().__init__()
        self.crop = sy, sx

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.crop]

    def invert_point(self, x: int, y: int) -> Tuple[float, float]:
        ys, xs = self.crop
        x += xs.start
        y += ys.start
        return x, y
