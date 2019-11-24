import numpy as np
import torch
import cv2


def images_to_batch(images, device=None, colors=3, model=None):
    """
    Converts OpenCV images to model input batch
    :param images: list of images with shape (height, width, color) or (height, width)
    :param device:
    :param color:
    :param model: Replaces device and color parameters, if specified
    :return: Resulting batch
    """
    if model:
        device = next(model.parameters()).device
        colors = model.n_channels

    # Set images shape as: (batch, height, width, color)
    images = np.array(images)
    if len(images.shape) == 3:
        images = np.expand_dims(images, -1)

    # Convert to desired colors
    shape = images.shape
    if shape[-1] != colors:
        if colors == 1 and shape[2] == 3:
            images = cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)
        elif colors == 3 and shape[2] == 1:
            images = cv2.cvtColor(images, cv2.COLOR_GRAY2RGB)

    # Convert to torch
    return torch.tensor(np.moveaxis(images, 3, 1), device=device, dtype=torch.float32)


def masks_to_batch(masks, device=None, classes=3, model=None, add_alpha=False):
    """
    Converts OpenCV masks images to model output batch
    :param masks: list of images with shape (height, width, color) or (height, width)
    :param device:
    :param color:
    :param model: Replaces device and color parameters, if specified
    :return: Resulting batch
    """
    if model:
        device = next(model.parameters()).device
        classes = model.n_classes

    # Set images shape as: (batch, height, width, color)
    masks = np.array(masks)
    if len(masks.shape) == 3:
        masks = np.expand_dims(masks, -1)

    # Convert to desired colors
    shape = masks.shape
    if shape[-1] != classes:
        if classes == 1 and shape[2] == 3:
            masks = cv2.cvtColor(masks, cv2.COLOR_RGB2GRAY)
        elif classes == 3 and shape[2] == 1:
            masks = cv2.cvtColor(masks, cv2.COLOR_GRAY2RGB)


    # Convert to torch
    return torch.tensor(np.moveaxis(masks, 3, 1), device=device, dtype=torch.float32) / 255
