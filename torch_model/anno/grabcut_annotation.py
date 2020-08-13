from typing import Optional, Tuple
import cv2
import numpy as np

from .annotation import Annotation


class GrabCutAnnotation(Annotation):
    def __init__(self, rect: Tuple[int, int, int, int]):
        super(GrabCutAnnotation, self).__init__()
        self.radius = 10
        self.mask = None
        self.gc_mask = None
        self.image = None
        self.rect = rect
        self.mode = cv2.GC_INIT_WITH_RECT
        self.draw_b = False
        self.draw_f = False
        self.initial_image = False
        self.bgd = np.zeros((1, 65), np.float64)
        self.fgd = np.zeros((1, 65), np.float64)
        self.se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process(self, frame: np.ndarray):
        pass

    def save(self, fn: str) -> bool:
        mask = self.gc_mask
        image = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.se)
        cv2.imwrite(fn + '.png', image)
        return True

    def visualize(self, image: np.ndarray, cursor: Optional[Tuple[int, int]], frame_no: int):
        if self.mask is None:
            self.mask = np.full(image.shape[:2], 128, np.uint8)
        self.image = image.copy()
        image[self.mask == 0] = 0
        image[self.mask == 255] |= 128
        if self.gc_mask is not None:
            # image[self.gc_mask == cv2.GC_PR_BGD, 2] = 255
            mask = np.where((self.gc_mask == 2) | (self.gc_mask == 0), 0, 255).astype('uint8')
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.se)
            contours = cv2.findContours(mask[:, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            cv2.drawContours(image, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)
        # image[:, :, 0] &= self.mask
        if cursor is not None:
            x, y = cursor
            cv2.circle(image, (x, y), self.radius, (0, 0, 0) if self.draw_b else (255, 255, 255), -1)

    def on_key(self, key) -> bool:
        if key == ord('g'):
            self.mask[:] = 128
            self.gc_mask = np.zeros_like(self.mask)
            cv2.grabCut(self.image, self.gc_mask, self.rect, self.bgd, self.fgd, 5, cv2.GC_INIT_WITH_RECT)
        elif key == ord('c'):
            self.gc_mask[self.mask == 255] = cv2.GC_FGD
            self.gc_mask[self.mask == 0] = cv2.GC_BGD
            self.gc_mask, self.bgd, self.fgd = cv2.grabCut(
                self.image, self.gc_mask, None, self.bgd, self.fgd, 5, cv2.GC_INIT_WITH_MASK
            )
        elif key == ord('='):
            self.radius += 1
        elif key == ord('-'):
            if self.radius > 1:
                self.radius -= 1
        else:
            return False
        return True

    def on_move(self, x: int, y: int):
        if self.draw_f:
            cv2.circle(self.mask, (x, y), self.radius, (255,), -1)
        elif self.draw_b:
            cv2.circle(self.mask, (x, y), self.radius, (0,), -1)

    def on_left_down(self, x: int, y: int):
        self.draw_f = True
        cv2.circle(self.mask, (x, y), self.radius, (255,), -1)

    def on_right_down(self, x: int, y: int):
        self.draw_b = True
        cv2.circle(self.mask, (x, y), self.radius, (0,), -1)

    def on_left_up(self, x: int, y: int):
        self.draw_f = False

    def on_right_up(self, x: int, y: int):
        self.draw_b = False
