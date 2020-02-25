from os import listdir
from os.path import isfile, isdir, join, splitext
from typing import Optional, Dict, Any

import cv2

from torch.utils.data import Dataset


class ParallelDataset(Dataset):
    def __init__(self,
                 root_dir='data',
                 classes=('images', 'masks'),
                 aug=None,
                 extensions=('.png', '.jpeg'),
                 loaders: Optional[Dict[str, Any]] = None,
                 post_proc=None
                 ):
        super().__init__()
        self.root_dir = root_dir
        self.loaders = loaders or {}
        self.classes = classes
        self.aug = aug or []
        self.items = []
        self.extensions = extensions
        self.post_proc = post_proc

    def find_item(self, fn):
        fn = splitext(fn)[0]
        for ex in self.extensions:
            if isfile(fn + ex):
                return fn + ex
        raise FileNotFoundError

    def update(self):
        self.items = [
            (join(self.root_dir, self.classes[0], fn),) + tuple(
                self.find_item(join(self.root_dir, cl, fn))
                for cl in self.classes[1:]
            )
            for fn in listdir(join(self.root_dir, self.classes[0]))
            if isfile(join(self.root_dir, self.classes[0], fn)) and any(map(fn.endswith, self.extensions))
        ]

    def __len__(self):
        self.update()
        return len(self.items)

    def __getitem__(self, item):
        items = [
            self.loaders.get(cl, cv2.imread)(fn)
            for cl, fn in zip(self.classes, self.items[item])
        ]
        for aug in self.aug:
            aug, image, mask = aug if len(aug) == 3 else (aug + (None,))
            image_idx = self.classes.index(image)
            mask_idx = None if mask is None else self.classes.index(mask)
            augmented = aug(
                image=items[image_idx],
                mask=items[mask_idx] if mask_idx is not None else None
            )
            items[image_idx] = augmented['image']
            if mask_idx is not None:
                items[mask_idx] = augmented['mask']
        if self.post_proc is not None:
            if type(self.post_proc) is dict:
                for name, proc in self.post_proc.items():
                    idx = self.classes.index(name)
                    items[idx] = proc(items[idx])
            else:
                items = self.post_proc(*items)
        return tuple(items)
