from typing import Iterable
from os import listdir
from os.path import isfile, isdir, join
from random import shuffle, randrange

import cv2
from torch.utils.data import IterableDataset, Dataset


class RandomSource:
    @staticmethod
    def get_files(directory, extensions: Iterable[str]):
        result = []
        for fn in listdir(directory):
            ffn = join(directory, fn)
            if any(map(fn.lower().endswith, extensions)) and isfile(ffn):
                result.append(fn)
            elif isdir(ffn):
                result += [join(fn, n) for n in RandomSource.get_files(ffn, extensions)]
        return result

    def __init__(self, directory, extensions: Iterable[str] = ('.png', '.mp4', '.mov')):
        self.directory = directory
        self.files = self.get_files(directory, extensions)
        self.idx = 0

    def __next__(self):
        if self.idx >= len(self.files):
            raise StopIteration
        fn = join(self.directory, self.files[self.idx])
        if any(map(fn.lower().endswith, ('.mp4', '.mov'))):
            cap = cv2.VideoCapture(fn)
            flag = False
            frame = None
            while not flag:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_number = randrange(frame_count)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                flag, frame = cap.read()
        else:
            frame = cv2.imread(fn)
        self.idx += 1
        return frame

    def frame_no(self) -> int:
        return self.idx

    def __iter__(self):
        shuffle(self.files)
        self.idx = 0
        return self

    def __len__(self) -> int:
        return len(self.files)


class PNGSource(Dataset):
    """Simple OpenCV source"""
    def __init__(self, directory, num_sort=True, sort_key=None, reverse=False):
        self.url = directory
        self.directory = directory
        files = [fn for fn in listdir(directory) if fn.endswith('.png') and isfile(join(directory, fn))]
        if num_sort and sort_key is None:
            self.files = sorted(files, key=lambda fn: int(fn[:-4]), reverse=reverse)
        else:
            self.files = sorted(files, key=sort_key, reverse=reverse)
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.files):
            raise StopIteration
        frame = cv2.imread(join(self.directory, self.files[self.idx]))
        self.idx += 1
        return frame

    def forward(self, frames=200):
        self.idx += frames

    def back(self, frames=1):
        self.idx -= frames + 1

    def frame_no(self) -> int:
        return self.idx

    def __len__(self) -> int:
        return len(self.files)


class OCVSource(IterableDataset):
    """Simple OpenCV source"""
    def __init__(self, url, clip: slice = None):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        if clip is None:
            clip = slice(0, None)
        self.clip = clip
        self.set_frame_no(0)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.clip.stop is not None and int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) >= self.clip.stop:
                raise StopIteration
            flag, frame = self.cap.read()
            if not flag:
                raise StopIteration
            return frame

    def __getitem__(self, item: int):
        self.set_frame_no(item)
        flag, frame = self.cap.read()
        if not flag:
            raise IndexError
        return frame

    def forward(self, frames=200):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) + frames)

    def back(self, frames=1):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) - frames - 1)

    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def frame_no(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - (self.clip.start or 0)

    def set_frame_no(self, frame_no: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no + (self.clip.start or 0))

    def __len__(self) -> int:
        c = self.clip
        if c.stop is not None:
            return c.stop - (c.start or 0)
        else:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
