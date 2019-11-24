import numpy as np
from torch_model.io import images_to_batch, batch_to_images, masks_to_batch, batch_to_masks


def test_image_io_rgb():
    images = np.random.randint(255, size=(4, 240, 320, 3), dtype=np.uint8)
    batch = images_to_batch(images, colors=3)
    print(batch.shape)
    images1 = batch_to_images(batch)
    assert images.dtype == images1.dtype
    assert images.shape == images1.shape
    assert np.all(images == images1)


def test_mask_io_rgb():
    masks = np.random.randint(255, size=(4, 240, 320, 3), dtype=np.uint8)
    batch = masks_to_batch(masks)
    print(batch.shape)
    masks_r = batch_to_masks(batch, threshold=None)
    assert masks.dtype == masks_r.dtype
    assert masks.shape == masks_r.shape
    assert np.all(masks == masks_r)
