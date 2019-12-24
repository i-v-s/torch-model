from os import listdir
from os.path import isfile, isdir, join
from copy import copy
from random import shuffle

import cv2

import numpy as np

import torch
from torch.utils.data import get_worker_info, IterableDataset


class VideoIterator:
    def __init__(self, items, directory, aug=None):
        self.items = items
        self.it = iter(items)
        self.aug = aug
        self.cap = None
        self.classes = None
        self.directory = directory
        self.open()

    def open(self):
        fn, self.classes = next(self.it)
        self.cap = cv2.VideoCapture(join(self.directory, fn))

    def __next__(self):
        flag, frame = self.cap.read()
        while not flag:
            self.open()
            flag, frame = self.cap.read()
        if self.aug:
            frame = self.aug(image=frame)['image']
        return np.moveaxis(frame, -1, 0), self.classes


class VideosDataset(IterableDataset):
    def __init__(self, directory, class_list, aug=None):
        super().__init__()
        self.directory = directory
        self.class_list = class_list #  or [name for name in listdir(directory) if isdir(join(directory, name))]
        self.class_map = {c: i for i, c in enumerate(self.class_list)}
        self.items = None
        self.aug = aug
        self.update()

    def classes_to_tensor(self, classes):
        result = np.zeros(len(self.class_list))
        for c in classes:
            result[self.class_map[c]] = 1
        return result

    def update(self):
        self.items = [
            (join(cn, fn), self.classes_to_tensor(cn.split('-')))
            for cn in listdir(self.directory) for fn in listdir(join(self.directory, cn))
            if cn in self.class_list and isfile(join(self.directory, cn, fn)) and (fn.endswith('.mp4') or fn.endswith('.avi'))
        ]

    def __iter__(self):
        info = get_worker_info()
        items = copy(self.items if info is None else self.items[info.id::info.num_workers])
        shuffle(items)
        return VideoIterator(items, self.directory, self.aug)
