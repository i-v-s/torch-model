from typing import Tuple
import numpy as np
import cv2
from math import pi, sin, cos
from .transform import AnnoTransform


class AffineTransform(AnnoTransform):
    def __init__(self, center: Tuple[float, float], out_size: Tuple[int, int],
                 angle: float, scale: float = 1.0):
        super(AffineTransform, self).__init__()
        angle *= (pi / 180)
        a, b = scale * cos(angle), scale * sin(angle)
        self.matrix = np.array([
            [a, b, out_size[0] / 2],
            [-b, a, out_size[1] / 2],
        ])
        self.matrix[:, 2] -= (self.matrix[:, :2] * np.array(center, dtype=np.float64)).sum(-1)
        self.out_size = out_size

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(frame, self.matrix, self.out_size)

    def invert_point(self, x: int, y: int) -> Tuple[float, float]:
        p = np.linalg.solve(self.matrix[:, :2], np.array([x, y], dtype=np.float64) - self.matrix[:, 2])
        return p.tolist()
