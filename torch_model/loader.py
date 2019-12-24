from typing import NamedTuple

import numpy as np

from torch import nn
import torch
from .blocks import Conv
import yaml

class LoaderInfo(NamedTuple):
    modules: dict
    values: dict

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(-1 if type(d) is str else d for d in shape)

    def forward(self, x):
        return x.reshape(self.shape)


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor):
        return x.permute(self.dims)


def permute(shape, *dims, batch_dims=1):
    assert len(shape) == len(dims) + batch_dims
    dims = tuple(range(batch_dims)) + tuple(d + batch_dims for d in dims)
    shape = tuple(shape[d] for d in dims)
    return shape, Permute(dims)


def conv2d(shape, out_channels, *args, **kwargs):
    n, c, h, w = shape
    conv = nn.Conv2d(c, out_channels, *args, **kwargs)
    h, w = (
        (v + 2 * conv.padding[i] - conv.dilation[i] * (conv.kernel_size[i] - 1) - 1) // conv.stride[i] + 1
        for i, v in enumerate((h, w))
    )
    return (n, out_channels, h, w), conv


def convT2d(shape, out_channels, *args, **kwargs):
    n, c, h, w = shape
    convT = nn.ConvTranspose2d(c, out_channels, *args, **kwargs)
    h, w = (
        (v - 1) * convT.stride[i] - 2 * convT.padding[i] +
        convT.dilation[i] * (convT.kernel_size[i] - 1) + convT.output_padding[i] + 1
        for i, v in enumerate((h, w))
    )
    return (n, out_channels, h, w), convT


def flatten(shape, batch_dims=1):
    shape = shape[:batch_dims] + (np.product(shape[batch_dims:]),)
    return shape, Reshape(shape)


def reshape(shape, *result_shape, batch_dims=1):
    assert np.product(shape[batch_dims:]) == np.product(result_shape)
    shape = shape[:batch_dims] + result_shape
    return shape, Reshape(shape)


def linear(shape, out_features, *args, **kwargs):
    return shape[:-1] + (out_features,), nn.Linear(shape[-1], out_features, *args, **kwargs)


def simple(module):
    def func(shape, *args, **kwargs):
        return shape, module(*args, **kwargs)
    return func


def bn2d(shape, *args, **kwargs):
    assert len(shape) == 4
    return shape, nn.BatchNorm2d(shape[1], *args, **kwargs)


def parse_params(params, values):
    args = []
    kwargs = {}
    allow_args = True
    for param in params if type(params) is list else [params]:
        if type(param) is dict:
            allow_args = False
            kwargs.update({k: values.get(v, v) if type(v) is str else v for k, v in param.items()})
        elif not allow_args:
            raise ValueError('Named parameters must be after positional')
        elif type(param) in [int, float]:
            args.append(param)
        elif type(param) is str:
            args.append(values.get(param, param))
        elif type(param) is list:
            args.append(tuple(param))
    return args, kwargs


def sequential(shape, items, li: LoaderInfo, verbose=True):
    if verbose:
        print('Sequence initial shape:', shape)
    result = []
    for item in items:
        if type(item) is str:
            shape, model = li.modules[item](shape)
            result.append(model)
        else:
            for name, params in item.items():
                args, kwargs = parse_params(params, li.values)
                shape, model = li.modules[name](shape, *args, **kwargs)
                if verbose:
                    title = f"After {name}({', '.join(map(str, args))}):"
                    title += ' ' * (32 - len(title))
                    print(f"{title} {shape}")
                result.append(model)
    return shape, nn.Sequential(*result)


modules = {
    'conv2d': conv2d, 'convt2d': convT2d, 'conv': Conv, 'seq': sequential,
    'flatten': flatten, 'reshape': reshape, 'permute': permute,
    'linear': linear,
    'relu': simple(nn.ReLU), 'prelu': simple(nn.PReLU), 'lrelu': simple(nn.LeakyReLU), 'sigmoid': simple(nn.Sigmoid),
    'dropout': simple(nn.Dropout), 'bn2d': bn2d
}


def load_yaml(fn):
    with open(fn, 'r') as file:
        conf = yaml.load(file)
    model_conf = conf['model']
    input_shape = tuple(model_conf['input'])
    classes = model_conf['classes']
    output_shape, model = modules['seq'](
        input_shape, model_conf['seq'],
        LoaderInfo(modules, {'classes': classes if type(classes) is int else len(classes)})
    )
    setattr(model, 'classes', model_conf['classes'])
    return model
