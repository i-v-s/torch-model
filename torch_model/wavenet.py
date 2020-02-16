import numpy as np
import torch.nn as nn
from torch import sigmoid, tanh, cat, zeros


def calc_l(L, conv):
    return (L + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1


class WaveStart(nn.Module):
    def __init__(self, in_channels, residual_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, residual_channels, 1, **kwargs)

    def forward(self, x):
        return self.conv(x).unsqueeze(-3), 0

    @classmethod
    def create(cls, shape, residual_channels, skip_channels, loader_info=None, **kwargs):
        # assert len(shape) == 3
        B = shape[:-2]
        C, L = shape[-2:]
        ws = WaveStart(C, residual_channels, **kwargs)
        L = calc_l(L, ws.conv)
        shape = B + (1, residual_channels, L)
        skip_shape = B + (skip_channels, L)
        return (shape, skip_shape), ws


class WaveCell(nn.Module):
    def __init__(self, x_channels, skip_channels, internal_channels, kernel_size, dilation, **kwargs):
        super().__init__()
        self.dilation = dilation
        self.filter_conv = nn.Conv1d(x_channels, internal_channels, kernel_size, **kwargs)
        self.gate_conv = nn.Conv1d(x_channels, internal_channels, kernel_size, **kwargs)
        self.residual_conv = nn.Conv1d(internal_channels, x_channels, 1)
        self.skip_conv = nn.Conv1d(internal_channels, skip_channels, 1)

    def dilate(self, x):
        if self.dilation == 1:
            return x
        D, C, L = x.shape[-3:]
        L_out = (L + self.dilation - 1) // self.dilation
        padding = L_out * self.dilation - L
        if padding > 0:
            x = cat([zeros(*x.shape[:-1], padding, dtype=x.dtype, device=x.device), x], -1)
        batch_len = len(x.shape) - 3
        x = x.permute(*range(batch_len), -2, -1, -3) # B, D, C, L -> B, C, L, D
        x = x.reshape(*x.shape[:-3], C, L_out, D * self.dilation)
        x = x.permute(*range(batch_len), -1, -3, -2) # B, C, L, D -> B, D, C, L
        return x

    def forward(self, inputs):
        x, skip = inputs
        residual = self.dilate(x)
        rs = residual.shape
        view = residual.reshape(np.product(rs[:-2]), rs[-2], rs[-1])
        filter = tanh(self.filter_conv(view))
        gate = sigmoid(self.gate_conv(view))
        x = gate * filter
        residual_inc = self.residual_conv(x).view(*(rs[:-1] + (-1,)))
        residual = residual[..., -residual_inc.shape[-1]:] + residual_inc
        skip_inc = self.skip_conv(x).view(*rs[:-2], -1, x.shape[-1]).permute(*range(len(rs) - 3), -2, -1, -3)
        skip_inc = skip_inc.reshape(*skip_inc.shape[:-2], -1)
        skip = skip_inc if type(skip) is int else skip[..., -skip_inc.shape[-1]:] + skip_inc
        # print(f'residual {residual.shape} skip {skip.shape}')
        return residual, skip

    @classmethod
    def create(cls, shape, channels, kernel_size, dilation=2, loader_info=None):
        x_shape, skip_shape = shape
        d, x_channels, x_l = x_shape[-3:]
        wc = WaveCell(x_channels, skip_shape[-2], channels, kernel_size, dilation)
        x_l = (x_l + dilation - 1) // dilation
        x_l = calc_l(calc_l(x_l, wc.gate_conv), wc.residual_conv)
        return (x_shape[:-3] + (d * dilation, x_channels, x_l), skip_shape[:-1] + (x_l * d * dilation,)), wc


class WaveReset(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        x, skip = inputs
        batch_len = len(x.shape) - 3
        D, C, L = x.shape[-3:]
        x = x.permute(*range(batch_len), -2, -1, -3) # B, D, C, L -> B, C, L, D
        x = x.reshape(*x.shape[:-3], C, -1, 1)
        x = x.permute(*range(batch_len), -1, -3, -2) # B, C, L, D -> B, D, C, L
        return x, skip

    @classmethod
    def create(cls, shape):
        x_shape, skip_shape = shape
        wr = WaveReset()
        return (x_shape[:-3] + (1, x_shape[-2], x_shape[-1] * x_shape[-3]), skip_shape), wr


class WaveEnd(nn.Module):
    def __init__(self, *args):
        super().__init__()
        convs = sum([[nn.ReLU(), nn.Conv1d(i, o, 1, bias=True)] for i, o in zip(args[:-1], args[1:])], [])
        self.seq = nn.Sequential(*convs)

    def forward(self, inputs):
        x, skip = inputs
        return self.seq(skip)

    @classmethod
    def create(cls, shape, *args, loader_info=None):
        x_shape, skip_shape = shape
        we = WaveEnd(skip_shape[-2], *args)
        return skip_shape[:-2] + (args[-1], skip_shape[-1]), we
