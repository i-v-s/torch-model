from typing import Optional, Tuple

import numpy as np

from torch_model import load_model


class Annotation:
    """Object of this class consists of annotation data and interface methods."""
    def __init__(self, model_name: Optional[str] = None, device=None, *args, **kwargs):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.player = None
        self.initial_image = True
        if model_name is not None:
            self.update_model(*args, **kwargs)

    def update_model(self, *args, **kwargs):
        self.model = load_model(self.model_name, device=self.device, *args, **kwargs)

    def process(self, frame: np.ndarray):
        raise NotImplementedError

    def set_player(self, player):
        self.player = player

    def clear(self):
        return

    def save(self, fn: str) -> bool:
        raise NotImplementedError

    def load(self, fn: str):
        ...

    def on_key(self, key) -> bool:
        return False

    def visualize(self, image: np.ndarray, cursor: Optional[Tuple[int, int]], frame_no: int):
        raise NotImplementedError

    def on_move(self, x: int, y: int):
        ...

    def on_left_down(self, x: int, y: int):
        ...

    def on_right_down(self, x: int, y: int):
        ...

    def on_left_up(self, x: int, y: int):
        ...

    def on_right_up(self, x: int, y: int):
        ...
