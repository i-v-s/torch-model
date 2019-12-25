import numpy as np
import torch


def batch_to_images(batch, to_np=True):
    batch = batch.clamp(0, 255).to(torch.uint8)
    return np.moveaxis(batch.cpu().numpy(), 1, -1) if to_np else batch


def batch_to_masks(batch, threshold=None, with_alpha=False):
    if with_alpha:
        batch = ((batch[:, :-1] > batch[:, -1:]) & (batch[:, :-1] > threshold)).to(torch.uint8) * 255
    else:
        batch = (batch * 255).to(torch.uint8) if threshold is None else ((batch > threshold).to(torch.uint8) * 255)
    return np.moveaxis(batch.cpu().numpy(), 1, -1)
