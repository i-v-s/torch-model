from copy import copy
from typing import NamedTuple

import numpy as np

from torch import nn
import torch
from .blocks import Conv
from .wavenet import WaveStart, WaveCell, WaveReset, WaveEnd
import yaml


class LoaderInfo(NamedTuple):
    modules: dict
    values: dict
    verbose: bool
    indent: str = ''


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


class Add(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.items = nn.ModuleList(modules)

    def forward(self, *args, **kwargs):
        result = self.items[0](*args, **kwargs)
        for item in self.items[1:]:
            result += item(*args, **kwargs)
        return result


def add_shape_calc(shapes):
    result = shapes[0]
    for shape in shapes[1:]:
        assert result == shape
    return result


class TupleProc(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.items = nn.ModuleList(modules)

    def forward(self, *input, **kwargs):
        print()


def tuple_shape_calc(shapes):
    print()


def permute(shape, *dims, batch_dims=1, loader_info=None):
    assert len(shape) == len(dims) + batch_dims
    dims = tuple(range(batch_dims)) + tuple(d + batch_dims for d in dims)
    shape = tuple(shape[d] for d in dims)
    return shape, Permute(dims)


def el(f, i):
    return f if type(f) is int else f[i]


def maxpool2d(shape, *args, loader_info=None, **kwargs):
    n, c, h, w = shape
    pool = nn.MaxPool2d(*args, **kwargs)
    h, w = (
        (v + 2 * el(pool.padding, i) - el(pool.dilation, i) * (el(pool.kernel_size, i) - 1) - 1) // el(pool.stride, i) + 1
        for i, v in enumerate((h, w))
    )
    return (n, c, h, w), pool


def maxpool3d(shape, *args, loader_info=None, **kwargs):
    n, c, d, h, w = shape
    pool = nn.MaxPool3d(*args, **kwargs)
    d, h, w = (
        (v + 2 * el(pool.padding, i) - el(pool.dilation, i) * (el(pool.kernel_size, i) - 1) - 1) // el(pool.stride, i) + 1
        for i, v in enumerate((d, h, w))
    )
    return (n, c, d, h, w), pool


def conv1d(shape, out_channels, *args, loader_info=None, **kwargs):
    n, c, l = shape
    conv = nn.Conv1d(c, out_channels, *args, **kwargs)
    l = (l + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
    return (n, out_channels, l), conv


def conv2d(shape, out_channels, *args, loader_info=None, **kwargs):
    n, c, h, w = shape
    conv = nn.Conv2d(c, out_channels, *args, **kwargs)
    h, w = (
        (v + 2 * conv.padding[i] - conv.dilation[i] * (conv.kernel_size[i] - 1) - 1) // conv.stride[i] + 1
        for i, v in enumerate((h, w))
    )
    return (n, out_channels, h, w), conv


def conv3d(shape, out_channels, *args, loader_info=None, **kwargs):
    n, c, d, h, w = shape
    conv = nn.Conv3d(c, out_channels, *args, **kwargs)
    d, h, w = (
        (v + 2 * conv.padding[i] - conv.dilation[i] * (conv.kernel_size[i] - 1) - 1) // conv.stride[i] + 1
        for i, v in enumerate((d, h, w))
    )
    return (n, out_channels, d, h, w), conv


def convT2d(shape, out_channels, *args, loader_info=None, **kwargs):
    n, c, h, w = shape
    convT = nn.ConvTranspose2d(c, out_channels, *args, **kwargs)
    h, w = (
        (v - 1) * convT.stride[i] - 2 * convT.padding[i] +
        convT.dilation[i] * (convT.kernel_size[i] - 1) + convT.output_padding[i] + 1
        for i, v in enumerate((h, w))
    )
    return (n, out_channels, h, w), convT


def convT1d(shape, out_channels, *args, loader_info=None, **kwargs):
    N, C, L = shape
    convT = nn.ConvTranspose1d(C, out_channels, *args, **kwargs)
    L = (
        (L - 1) * convT.stride[0] - 2 * convT.padding[0] +
        convT.dilation[0] * (convT.kernel_size[0] - 1) + convT.output_padding[0] + 1
    )
    return (N, out_channels, L), convT


def flatten(shape, batch_dims=1, loader_info=None):
    shape = shape[:batch_dims] + (np.product(shape[batch_dims:]),)
    return shape, Reshape(shape)


def reshape(shape, *result_shape, batch_dims=1, loader_info=None):
    assert np.product(shape[batch_dims:]) == np.product(result_shape)
    shape = shape[:batch_dims] + result_shape
    return shape, Reshape(shape)


def linear(shape, out_features, *args, loader_info=None, **kwargs):
    return shape[:-1] + (out_features,), nn.Linear(shape[-1], out_features, *args, **kwargs)


class Aggregate(nn.Module):
    def __init__(self, func, dim):
        super().__init__()
        self.dim = dim
        self.func = func

    def forward(self, x):
        result = self.func(x, dim=self.dim)
        if self.func is torch.max:
            return result[0]
        return result

    @classmethod
    def create(cls, func):
        def wrapper(shape, dim, loader_info=None):
            shape = list(shape)
            for d in [dim] if type(dim) is int else dim:
                shape[d] = None
            return tuple(filter(lambda d: d is not None, shape)), cls(func, dim)
        return wrapper


def simple(module):
    def func(shape, *args, loader_info=None, **kwargs):
        return shape, module(*args, **kwargs)
    return func


def nop(shape, *args, loader_info=None, **kwargs):
    return shape, nn.Identity(*args, **kwargs)


def bn2d(shape, *args, loader_info=None, **kwargs):
    assert len(shape) == 4
    return shape, nn.BatchNorm2d(shape[1], *args, **kwargs)


def bn3d(shape, *args, loader_info=None, **kwargs):
    assert len(shape) == 5
    return shape, nn.BatchNorm3d(shape[1], *args, **kwargs)


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


def print_item(indent, name, shape, args=None):
    title = f'{indent}{name}'
    if args:
        title += f"({', '.join(map(str, args))})"
    print(f'{title:50} {shape}')


def container(module, module_name, shape_calc=None):
    def wrapper(shape, items, loader_info: LoaderInfo, seq_name=''):
        modules, values, verbose, indent = loader_info
        if verbose:
            print_item(indent, f'{module_name} {seq_name}', shape)
        indent += '  '
        loader_info = LoaderInfo(modules, values, verbose, indent)
        result, shapes = [], []
        for item in items:
            if type(item) is str:
                new_shape, model = loader_info.modules[item](shape)
                if verbose:
                    print_item(indent, item, shape if new_shape != shape else '')
                result.append(model)
                if shape_calc:
                    shapes.append(new_shape)
                else:
                    shape = new_shape
            elif type(item) == dict and next(iter(item.keys())) in ['seq', 'add']:
                item = list(item.items())
                assert len(item) == 1
                key, sub_items = item[0]
                new_shape, model = modules[key](shape, sub_items, loader_info)
                result.append(model)
                if shape_calc:
                    shapes.append(new_shape)
                else:
                    shape = new_shape
            else:
                for name, params in item.items():
                    args, kwargs = parse_params(params, values)
                    new_shape, model = modules[name](shape, *args, **kwargs, loader_info=loader_info)
                    if verbose:
                        print_item(indent, name, new_shape, args)
                        # title = f"{indent}{name}({', '.join(map(str, args))}):"
                        # print(f"{title:50} {shape}")
                    result.append(model)
                    if shape_calc:
                        shapes.append(new_shape)
                    else:
                        shape = new_shape
        return shape_calc(shapes) if shape_calc else shape, module(*result)
    return wrapper


global_modules = {
    'conv1d': conv1d, 'conv2d': conv2d, 'conv3d': conv3d, 'convt1d': convT1d, 'convt2d': convT2d,
    'conv': Conv, 'maxpool2d': maxpool2d, 'mp2d': maxpool2d, 'maxpool3d': maxpool3d, 'mp3d': maxpool3d,
    'flatten': flatten, 'reshape': reshape, 'permute': permute,
    'linear': linear, 'max': Aggregate.create(torch.max),
    'relu': simple(nn.ReLU), 'prelu': simple(nn.PReLU), 'lrelu': simple(nn.LeakyReLU), 'sigmoid': simple(nn.Sigmoid),
    'dropout': simple(nn.Dropout), 'do2d': simple(nn.Dropout2d), 'do3d': simple(nn.Dropout3d),
    'bn2d': bn2d, 'bn3d': bn3d,
    'nop': nop, 'add': container(Add, 'add', add_shape_calc), 'seq': container(nn.Sequential, 'seq'),
    'tuple': container(TupleProc, 'tuple', tuple_shape_calc),
    'wn_start': WaveStart.create, 'wn_cell': WaveCell.create, 'wn_reset': WaveReset.create, 'wn_end': WaveEnd.create
}


def module_dec(name, module):
    params = module.get('params', [])

    def wrapper(shape, *args, loader_info: LoaderInfo = None):
        modules, values, verbose, indent = loader_info
        values = copy(values)
        values.update({k: v for k, v in zip(params, args)})
        for key in ['seq', 'add']:
            if key in module:
                return modules[key](
                    shape, module[key],
                    LoaderInfo(modules, values, verbose, indent),
                    seq_name=name
                )
    return wrapper


def update_modules(modules, new_modules):
    for name, item in new_modules.items():
        modules[name] = module_dec(name, item)


def load_yaml(fn, verbose=True, **params):
    with open(fn, 'r') as file:
        conf = yaml.safe_load(file)
    modules = copy(global_modules)
    if 'modules' in conf:
        update_modules(modules, conf['modules'])
    model_conf = conf['model']
    params = params.copy()
    classes = model_conf['classes']
    params['classes'] = classes if type(classes) is int else len(classes)
    input_shape = tuple(params.get(i, i) for i in model_conf['input'])
    output_shape, model = modules['seq'](
        input_shape, model_conf['seq'],
        LoaderInfo(modules, params, verbose),
        seq_name=''
    )
    setattr(model, 'classes', model_conf['classes'])
    return model
