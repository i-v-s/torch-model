import numpy as np
import cv2

from os import mkdir
from os.path import join, isdir
from shutil import rmtree

from .gen import png_pair_gen
from .io import images_to_batch

class Aug:
    def aug(self, image, mask):
        return {'image': image, 'mask': mask}


def test_png_pair_gen():
    image = np.random.randint(255, size=(240, 320, 3), dtype=np.uint8)
    mask = np.random.randint(2, size=(240, 320, 3), dtype=np.uint8) * 255
    test_dir = 'temp/ppg'
    if isdir('temp'):
        rmtree('temp')
    mkdir('temp')
    mkdir(test_dir)
    images_dir, masks_dir = join(test_dir, 'images'), join(test_dir, 'masks')
    mkdir(images_dir)
    mkdir(masks_dir)
    cv2.imwrite(join(images_dir, '0001.png'), image)
    cv2.imwrite(join(masks_dir, '0001.png'), mask)
    aug = Aug()
    gen = png_pair_gen(1, aug, images_dir, masks_dir, colors=3, classes=3)
    image_r, m = next(gen)
    image_b = images_to_batch([image])
    assert image_b.dtype == image_r.dtype
    assert image_b.shape == image_r.shape
    assert (image_b == image_r).all()
