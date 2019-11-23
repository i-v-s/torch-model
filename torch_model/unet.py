import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, layers=2):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            *sum([[
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ] for i in range(layers)], [])
        )

    def forward(self, x):
        try:
            x = self.conv(x)
        except RuntimeError as e:
            raise e
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, layers=2):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            Conv(in_ch, out_ch, layers)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, layers=2, bilinear=True):
        super(Up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = Conv(in_ch, out_ch, layers)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


UNetLayerInfo = namedtuple('UNetLayerInfo', 'down_channels down_layers up_channels up_layers')
DownInfo = namedtuple('DownInfo', 'in_ch out_ch layers')
UpInfo = namedtuple('UpInfo', 'in_ch out_ch layers')
classes = {'conv2d': nn.Conv2d, 'conv': Conv, 'relu': nn.ReLU}


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, downs, ups, pre=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.reduce = self.calc_reduce(pre)
        self.inc = nn.Sequential(*[
            classes[layer[0].lower()](*layer[1:])
            for layer in pre
        ]) if pre else Conv(n_channels, downs[0][0])

        self.downs = nn.ModuleList([Down(pr_c, nx_c, pr_l) for (pr_c, pr_l), (nx_c, _) in zip(downs[:-1], downs[1:])])

        self.ups = nn.ModuleList([
            Up(downs[i][0] + (ups[i + 1][0] if i < len(ups) - 1 else downs[-1][0]), out_c, layers)
            for i, (out_c, layers) in enumerate(ups)
        ])
        self.outc = nn.Conv2d(ups[0][0], n_classes, 1)

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
        params = self.outc.state_dict()
        add = n_classes - self.outc.out_channels
        bias = params['bias']
        params['bias'] = torch.cat([bias[:-1], torch.randn(add) * var, bias[-1:]])
        weight = params['weight']
        params['weight'] = torch.cat([weight[:-1], torch.randn(add, *weight.shape[1:]) * var, weight[-1:]])
        outc = nn.Conv2d(self.outc.in_channels, n_classes, 1)
        outc.load_state_dict(params)
        self.outc = outc

    def forward(self, x):
        xs = [None] * (len(self.downs) + 1)
        xs[0] = self.inc(x)
        for i, down in enumerate(self.downs):
            xs[i + 1] = down(xs[i])
        x = xs[-1]
        for i, up in reversed(list(enumerate(self.ups))):
            x = up(x, xs[i])

        x = self.outc(x)
        return torch.sigmoid(x)
