from os import listdir
from os.path import isfile, isdir, join

import cv2

import numpy as np

import torch
from torch.utils.data import Dataset


class ClassDirsDataset(Dataset):
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
            if isfile(join(self.directory, cn, fn)) and (fn.endswith('.png') or fn.endswith('.jpeg'))
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        fn, classes = self.items[item]
        image = cv2.imread(join(self.directory, fn))
        if self.aug:
            image = self.aug(image=image)['image']
        return np.moveaxis(image, -1, 0), classes
