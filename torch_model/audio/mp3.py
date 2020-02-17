from os import listdir
from os.path import join, isfile
import numpy as np
import pydub

from torch.utils.data import get_worker_info, IterableDataset, Dataset


class CachedSample:
    def __init__(self, file_name: str, window, step, name=None):
        """MP3 to numpy array"""
        a = pydub.AudioSegment.from_mp3(file_name)
        y = np.array(a.get_array_of_samples())
        if a.channels == 2:
            y = y.reshape((-1, 2))
        self.data = y # a.frame_rate,
        self.step = step
        self.window = window
        self.steps = (y.shape[0] - window) // step
        self.name = name
        self.normalize = True

    def __getitem__(self, item):
        item *= self.step
        result = self.data[item : item + self.window].astype(np.float32)
        if self.normalize:
            result /= 32768
        return result if self.name is None else (self.name, result)

class CachedAudioDataset(Dataset):
    def load_dir(self, directory):
        return [
            CachedSample(join(directory, fn), self.window, self.step)
            for fn in listdir(directory)
            if isfile(join(directory, fn)) and fn.endswith('.mp3')
        ]

    def __init__(self, directory, window=32768, step=1024, classes=None):
        self.window = window
        self.step = step
        if classes is None:
            self.items = self.load_dir(directory)
        self.total_steps = 0
        for item in self.items:
            self.total_steps += item.steps

    def __len__(self):
        return self.total_steps

    def __getitem__(self, index):
        for item in self.items:
            if index < item.steps:
                return item[index]
        raise IndexError
