from os import listdir
from os.path import isfile, join
from random import shuffle, randrange

import cv2


class RandomSource:
    def __init__(self, directory, extensions=('.png', '.mp4')):
        self.directory = directory
        self.files = [
            fn
            for fn in listdir(directory)
            if any(map(fn.endswith, extensions)) and isfile(join(directory, fn))
        ]
        self.idx = 0

    def __next__(self):
        if self.idx >= len(self.files):
            raise StopIteration
        fn = join(self.directory, self.files[self.idx])
        if fn.endswith('.mp4'):
            cap = cv2.VideoCapture(fn)
            frame_number = randrange(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
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


class PNGSource:
    """Simple OpenCV source"""
    def __init__(self, directory, num_sort=True):
        self.url = directory
        self.directory = directory
        files = [fn for fn in listdir(directory) if fn.endswith('.png') and isfile(join(directory, fn))]
        if num_sort:
            self.files = sorted(files, key=lambda fn: int(fn[:-4]))
        else:
            self.files = sorted(files)
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


class OCVSource:
    """Simple OpenCV source"""
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            flag, frame = self.cap.read()
            if not flag:
                raise StopIteration
            return frame

    def forward(self, frames=200):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) + frames)

    def back(self, frames=1):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) - frames - 1)

    def frame_no(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def set_frame_no(self, frame_no: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    def __len__(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
