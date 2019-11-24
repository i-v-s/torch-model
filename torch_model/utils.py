import cv2
import json
from os.path import join, isfile, isdir
from os import mkdir
from typing import Tuple
from . import UNet
import numpy as np
import torch
from torch.nn import Module
from torch.optim import Adam, SGD, RMSprop, Adagrad, Adadelta
from .io import batch_to_masks, images_to_batch

models = {'unet': UNet}
optimizers = {
    'adam': Adam,
    'sgd': SGD,
    'adagrad': Adagrad,
    'adadelta': Adadelta,
    'rmsprop': RMSprop
}


def load_model(name, train=False, directory='models', device=None, best=False, n_classes=None):
    model_dir = join(directory, name)
    if not isdir(model_dir):
        mkdir(model_dir)
    with open(join('models', '%s.json' % name)) as f:
        params = json.load(f)
    model_type = params['type']
    del params['type']
    model = models[model_type](**params)
    weights_fn = join(model_dir, ('%s_best.pt' if best else '%s.pt') % name)
    if not isfile(weights_fn):
        weights_fn = join(directory, ('%s_best.pt' if best else '%s.pt') % name)
    if isfile(weights_fn):
        model.load_state_dict(torch.load(weights_fn))
    if n_classes is not None and n_classes > params['n_classes']:
        model.expand_out(n_classes)
    model = model.to(device)
    if train:
        model.train()
    else:
        model.eval()
    if isfile(join(model_dir, 'state.json')):
        with open(join(model_dir, 'state.json'), 'r') as f:
            state = json.load(f)
        return model, state.get('best_loss', None), state.get('epoch', 1)
    else:
        return model, None, 1


def save_model(name, model: Module, best_loss, epoch, directory='models'):
    torch.save(model.state_dict(), join(directory, name, '%s.pt' % name))
    with open(join(directory, name, 'state.json'), 'w') as f:
        json.dump({'best_loss': best_loss, 'epoch': epoch}, f, indent='  ')


def get_device():
    print('PyTorch version:', torch.__version__)
    use_cuda = torch.cuda.is_available()
    print('Use CUDA:', torch.version.cuda if use_cuda else False)
    return torch.device('cuda' if use_cuda else 'cpu')


def create_optimizer(config, model):
    return optimizers[config['type']](model.parameters(), **{k: v for k, v in config.items() if k != 'type'})


def process_image(frame, model, threshold=0.5):
    batch = images_to_batch([frame], model=model)
    mask = model(batch)
    return np.squeeze(batch_to_masks(mask, threshold), 0)


def get_process(model, threshold=0.5):
    def process(frame):
        return process_image(frame, model, threshold)
    return process


def scale_with_padding(target_shape: Tuple[int, int, int], source: np.ndarray,
                       scale: int, interpolation=cv2.INTER_NEAREST):
    h, w, c = source.shape
    w, h = w * scale, h * scale
    source = cv2.resize(source, (w, h), interpolation=interpolation)
    fh, fw = target_shape[0], target_shape[1]
    dw, dh = fw - w, fh - h
    dtype = source.dtype
    source = np.concatenate([
        np.zeros((dh // 2, w, c), dtype=dtype),
        source,
        np.zeros((dh - dh // 2, w, c), dtype=dtype)
    ], 0)
    source = np.concatenate([
        np.zeros((fh, dw // 2, c), dtype=dtype),
        source,
        np.zeros((fh, dw - dw // 2, c), dtype=dtype)
    ], 1)
    return source


def un_scale(target_shape: Tuple[int, int, int], source: np.ndarray, scale: int):
    th, tw, tc = target_shape
    sh, sw, sc = source.shape
    sx, sy = (sw - tw * scale) // 2, (sh - th * scale) // 2
    source = source[sy:sy + th * scale, sx:sx + tw * scale]
    return cv2.resize(source, (tw, th))
