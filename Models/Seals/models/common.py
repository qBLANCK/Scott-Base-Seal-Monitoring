import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.tools import Struct


def identity(x, **kwargs):
    return x


def reverse(xs):
    return list(reversed(xs))


# @torch.jit.script
def match_size_2d(t, w: int, h: int):
    dh = h - t.shape[2]
    dw = w - t.shape[3]

    pad = (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)
    return F.pad(t, pad)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def cascade(modules, input):
    outputs = []

    for module in modules:
        input = module(input)
        outputs.append(input)

    return outputs


class Cascade(nn.Sequential):
    def __init__(self, *args, drop_initial=0):
        super(Cascade, self).__init__(*args)
        self.drop = drop_initial

    def forward(self, input):
        out = cascade(self._modules.values(), input)
        return out[self.drop:]


class UpCascade(nn.Module):
    def __init__(self, *decoders):
        super(UpCascade, self).__init__()

        self.decoders = nn.Sequential(*decoders)

    def forward(self, inputs):
        assert len(inputs) == len(self.decoders)

        input = None
        outputs = []

        for module, skip in zip(
                reverse(self.decoders._modules.values()), reverse(inputs)):
            input = module(input, skip)
            outputs.append(input)

        return reverse(outputs)


class Parallel(nn.Module):
    def __init__(self, *modules):
        super(Parallel, self).__init__()
        self.parallel = nn.Sequential(*modules)

    def forward(self, inputs):
        assert len(inputs) == len(self.parallel)
        assert isinstance(
            inputs, list), "type of inputs is: " + str(type(inputs))

        return [m(i) for m, i in zip(self.parallel, inputs)]


class Named(nn.Module):
    def __init__(self, **named):
        super(Named, self).__init__()

        for k, v in named.items():
            self.add_module(k, v)

    def forward(self, input):
        output = {k: module(input) for k, module in self.named_children()}
        return Struct(output)


class Residual(nn.Sequential):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        output = self.module(input)
        return output + input


class Lookup(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, inputs):
        return inputs[self.k]


class Conv(nn.Module):
    def __init__(
            self,
            in_size,
            out_size,
            kernel=3,
            stride=1,
            padding=None,
            bias=False,
            activation=nn.ReLU(
                inplace=True),
            groups=1):
        super().__init__()

        padding = kernel // 2 if padding is None else padding

        self.norm = nn.BatchNorm2d(in_size)
        self.conv = nn.Conv2d(
            in_size,
            out_size,
            kernel,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=1)
        self.activation = activation

    def forward(self, inputs):
        return self.conv(self.activation(self.norm(inputs)))


class Deconv(nn.Module):
    def __init__(
            self,
            in_size,
            out_size,
            kernel=3,
            stride=1,
            padding=None,
            bias=False,
            activation=nn.ReLU(
                inplace=True),
            groups=1):
        super().__init__()

        padding = kernel // 2 if padding is None else padding

        self.norm = nn.BatchNorm2d(in_size)
        self.conv = nn.ConvTranspose2d(
            in_size,
            out_size,
            kernel,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=1)
        self.activation = activation

    def forward(self, inputs):
        return self.conv(self.activation(self.norm(inputs)))


def basic_block(in_size, out_size):
    return nn.Sequential(
        Conv(in_size, out_size, activation=identity),
        Conv(out_size, out_size),
        nn.BatchNorm2d(out_size)
    )


class Upscale(nn.Module):
    def __init__(self, features, scale_factor=2):
        super().__init__()
        self.conv = Conv(features, features * scale_factor ** 2)
        self.scale_factor = scale_factor

    def forward(self, inputs):
        return F.pixel_shuffle(self.conv(inputs), self.scale_factor)


def make_upscale(features, scale_factor, method):
    if method in ['nearest', 'linear', 'bilinear']:
        return nn.Upsample(scale_factor=scale_factor, mode=method)
    elif method == 'shuffle':
        return Upscale(features, scale_factor=scale_factor)
    elif method == 'conv':
        return Deconv(features, features, stride=2, padding=0)
    else:
        assert False, "unknown upscale method: " + method


class Decode(nn.Module):
    def __init__(self, features, module=None,
                 scale_factor=2, upscale='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.reduce = Conv(features * 2, features)
        self.module = module or identity
        self.upscale = make_upscale(features, scale_factor, method=upscale)

    def forward(self, inputs, skip):
        if not (inputs is None):
            upscaled = self.upscale(inputs)
            trim = match_size_2d(upscaled, skip.shape[3], skip.shape[2])
            return self.module(self.reduce(torch.cat([trim, skip], 1)))

        return self.module(skip)
