from os import listdir
from os.path import isfile, join
from random import choice
import numpy as np
import torch
import cv2
from .utils import scale_with_padding, un_scale


def png_pair_gen(batch, aug, images='images', masks='masks', device=None, colors=1, classes=4, scale=1):
    c, h, w = aug.shape
    image_batch = np.empty((batch, colors, h, w), dtype=np.float32)
    mask_batch = None
    three_batch = np.empty((batch, classes, 1, 1), dtype=np.float32)
    alpha = classes - 1
    while True:
        names = listdir(images)
        for b in range(batch):
            name = choice(names)
            image = cv2.imread(join(images, name))
            mask = cv2.imread(join(masks, name))
            mask_size = mask.shape
            if scale != 1:
                mask = scale_with_padding(image.shape, mask, scale)
            three = np.ones((classes, 1, 1), dtype=np.float32)
            data = aug.aug(image=image, mask=mask)
            mask = un_scale(mask_size, data['mask'], scale) if scale != 1 else data['mask']
            image = np.moveaxis(data['image'], -1, 0).astype(np.float32) / 255
            mask = np.moveaxis(mask, -1, 0).astype(np.float32) / 255
            if image.shape[0] == 3 and colors == 1:
                image = np.mean(image, axis=0, keepdims=True)
            image_batch[b] = image
            if mask_batch is None:
                mask_batch = np.empty((batch, classes) + mask_size[:2], dtype=np.float32)
            mask_batch[b, :alpha] = mask * three[:alpha]
            mask_batch[b, alpha] = 1.0 - np.max(mask[:alpha], 0)
            three_batch[b] = three
        yield (
            torch.tensor(image_batch, device=device),
            torch.tensor(mask_batch, device=device),
            torch.tensor(three_batch, device=device)
        )
