import torch
from torch import nn
from collections import namedtuple
from .blocks import Conv, Up, Down


UNetLayerInfo = namedtuple('UNetLayerInfo', 'down_channels down_layers up_channels up_layers')
DownInfo = namedtuple('DownInfo', 'in_ch out_ch layers')
UpInfo = namedtuple('UpInfo', 'in_ch out_ch layers')
classes = {'conv2d': nn.Conv2d, 'conv': Conv, 'relu': nn.ReLU}
out_layers = {'sigmoid': lambda: nn.Sigmoid(), 'softmax': lambda: nn.Softmax(1)}


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, downs, ups, pre=None, with_alpha=False, out_activation='sigmoid', lrelu=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.with_alpha = with_alpha
        self.out_activation = out_activation

        self.reduce = self.calc_reduce(pre)
        self.inc = nn.Sequential(*[
            classes[layer[0].lower()](*layer[1:])
            for layer in pre
        ]) if pre else Conv(n_channels, downs[0][0], lrelu=lrelu)

        self.downs = nn.ModuleList([
            Down(pr_c, nx_c, pr_l, lrelu) for (pr_c, pr_l), (nx_c, _) in zip(downs[:-1], downs[1:])
        ])

        self.ups = nn.ModuleList([
            Up(downs[i][0] + (ups[i + 1][0] if i < len(ups) - 1 else downs[-1][0]), out_c, layers, lrelu=lrelu)
            for i, (out_c, layers) in enumerate(ups)
        ])
        self.out_c = nn.Conv2d(ups[0][0], n_classes, 1)
        self.out_a = out_layers[self.out_activation]() if self.out_activation is not None else None

    @staticmethod
    def calc_reduce(pre):
        padding, scale = 0, 1
        if pre:
            for layer in pre:
                if layer[0].lower() == 'conv2d':
                    kernel, stride = (layer + [1])[3:5]
                    padding *= stride
                    scale *= stride
                    padding += (kernel - 1) // 2
        return padding, scale

    def expand_out(self, n_classes, var=0.1):
        params = self.out_c.state_dict()
        add = n_classes - self.out_c.out_channels
        bias = params['bias']
        params['bias'] = torch.cat([bias[:-1], torch.randn(add) * var, bias[-1:]])
        weight = params['weight']
        params['weight'] = torch.cat([weight[:-1], torch.randn(add, *weight.shape[1:]) * var, weight[-1:]])
        outc = nn.Conv2d(self.out_c.in_channels, n_classes, 1)
        outc.load_state_dict(params)
        self.out_c = outc

    def forward(self, x):
        xs = [None] * (len(self.downs) + 1)
        xs[0] = self.inc(x)
        for i, down in enumerate(self.downs):
            xs[i + 1] = down(xs[i])
        x = xs[-1]
        for i, up in reversed(list(enumerate(self.ups))):
            x = up(x, xs[i])

        x = self.out_c(x)
        return x if self.out_a is None else self.out_a(x)
