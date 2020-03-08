from os import listdir
from os.path import isfile, isdir, join, splitext
from typing import Optional, Dict, Any, Tuple
from fractions import Fraction

import cv2

from torch.utils.data import Dataset


class ParallelDataset(Dataset):
    def __init__(self,
                 root_dir='data',
                 classes=('images', 'masks'),
                 aug=None,
                 extensions=('.png', '.jpeg'),
                 loaders: Optional[Dict[str, Any]] = None,
                 post_proc=None,
                 file_filter=None
                 ):
        super().__init__()
        self.root_dir = root_dir
        self.loaders = loaders or {}
        self.classes = classes
        self.aug = aug or []
        self.items = []
        self.extensions = extensions
        self.post_proc = post_proc
        self.file_filter = file_filter

    @classmethod
    def create_pair(cls, *args,
                    aug=None, eval_aug=None,
                    fraction: Fraction = Fraction(1, 7),
                    **kwargs) -> Tuple[Dataset, Dataset]:
        """
        Divides all samples by two datasets: training and evaluation. Disables augmentation for evaluation
        :param aug:
        :param args:
        :param fraction: Part of evaluation data
        :param kwargs:
        :return:
        """
        return (
            cls(*args, **kwargs, aug=aug, file_filter=lambda i, f: i % fraction.denominator >= fraction.numerator),
            cls(*args, **kwargs, aug=eval_aug, file_filter=lambda i, f: i % fraction.denominator < fraction.numerator)
        )

    def find_item(self, fn):
        fn = splitext(fn)[0]
        for ex in self.extensions:
            if isfile(fn + ex):
                return fn + ex
        raise FileNotFoundError

    def update(self):
        files = (
            fn
            for fn in sorted(listdir(join(self.root_dir, self.classes[0])))
            if isfile(join(self.root_dir, self.classes[0], fn)) and any(map(fn.endswith, self.extensions))
        )
        if self.file_filter is not None:
            files = (f for i, f in enumerate(files) if self.file_filter(i, f))
        self.items = [
            (join(self.root_dir, self.classes[0], fn),) + tuple(
                self.find_item(join(self.root_dir, cl, fn))
                for cl in self.classes[1:]
            )
            for fn in files
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
