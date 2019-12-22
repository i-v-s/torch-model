from os import listdir
from os.path import join
from random import choice
import numpy as np
import torch
import cv2
from .utils import scale_with_padding, un_scale
from .io import images_to_batch, masks_to_batch



def png_pair_gen(batch, aug, images_dir='images', masks_dir='masks', device=None, colors=1, classes=4, scale=1):
    # c, h, w = aug.shape
    three_batch = np.empty((batch, classes, 1, 1), dtype=np.float32)
    while True:
        names = listdir(images_dir)
        images = []
        masks = []
        for b in range(batch):
            name = choice(names)
            image = cv2.imread(join(images_dir, name))
            mask = cv2.imread(join(masks_dir, name))
            mask_size = mask.shape
            if scale != 1:
                mask = scale_with_padding(image.shape, mask, scale)
            data = aug.aug(image=image, mask=mask)
            mask = un_scale(mask_size, data['mask'], scale) if scale != 1 else data['mask']

            images.append(data['image'])
            masks.append(mask)

            # image = np.moveaxis(data['image'], -1, 0).astype(np.float32) / 255
            # mask = np.moveaxis(mask, -1, 0).astype(np.float32) / 255
            # if image.shape[0] == 3 and colors == 1:
            #     image = np.mean(image, axis=0, keepdims=True)
            # image_batch[b] = image
            # if mask_batch is None:
            #     mask_batch = np.empty((batch, classes) + mask_size[:2], dtype=np.float32)
            # mask_batch[b, :alpha] = mask * three[:alpha]
            # mask_batch[b, alpha] = 1.0 - np.max(mask[:alpha], 0)
            # three_batch[b] = three
        yield (
            images_to_batch(images, device, colors),
            masks_to_batch(masks, device, classes)
        )
