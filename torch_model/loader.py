from typing import NamedTuple

import numpy as np

from torch import nn
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


def conv2d(shape, out_channels, *args, **kwargs):
    n, c, h, w = shape
    conv = nn.Conv2d(c, out_channels, *args, **kwargs)
    h, w = (
        (v + 2 * conv.padding[i] - conv.dilation[i] * (conv.kernel_size[i] - 1) - 1) // conv.stride[i] + 1
        for i, v in enumerate((h, w))
    )
    return (n, out_channels, h, w), conv


def flatten(shape, batch_dims=1):
    shape = shape[:batch_dims] + (np.product(shape[batch_dims:]),)
    return shape, Reshape(shape)

def linear(shape, out_features, *args, **kwargs):
    return shape[:-1] + (out_features,), nn.Linear(shape[-1], out_features, *args, **kwargs)


def simple(module):
    def func(shape, *args, **kwargs):
        return shape, module(*args, **kwargs)
    return func


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
    return args, kwargs


def sequential(shape, items, li: LoaderInfo):
    result = []
    for item in items:
        if type(item) is str:
            shape, model = li.modules[item](shape)
            result.append(model)
        else:
            for name, params in item.items():
                args, kwargs = parse_params(params, li.values)
                shape, model = li.modules[name](shape, *args, **kwargs)
                result.append(model)
    return shape, nn.Sequential(*result)


modules = {
    'conv2d': conv2d, 'conv': Conv, 'seq': sequential, 'flatten': flatten, 'linear': linear,
    'relu': simple(nn.ReLU), 'sigmoid': simple(nn.Sigmoid), 'dropout': simple(nn.Dropout)
}


def load_yaml(fn):
    with open(fn, 'r') as file:
        conf = yaml.load(file)
    model_conf = conf['model']
    input_shape = tuple(model_conf['input'])
    output_shape, model = modules['seq'](
        input_shape, model_conf['seq'],
        LoaderInfo(modules, {'classes': len(model_conf['classes'])})
    )
    setattr(model, 'classes', model_conf['classes'])
    return model
