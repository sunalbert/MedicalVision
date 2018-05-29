import torch
import torch.nn as nn


class ConvBnRelu2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, is_bn=True,
                 is_relu=True):
        super(ConvBnRelu2D, self).__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self._bn = nn.BatchNorm2d(out_channels)
        self._relu = nn.ReLU(inplace=True)

        if is_bn is False:
            self._bn = None

        if is_relu is False:
            self._relu = None

    def forward(self, x):
        x = self._conv(x)
        if self._bn:
            x = self._bn(x)
        if self._relu:
            x = self._relu(x)
        return x

    def merge_bn(self):
        if self._bn is None:
            return

        assert self._conv.bias is None

        conv_weight = self._conv.weight.data
        bn_gamma = self._bn.weight.data
        bn_beta = self._bn.bias.data
        running_mean = self._bn.running_mean
        running_var = self._bn.running_var
        eps = self._bn.eps

        out_planes, in_planes, h, w = conv_weight.size()
        std = 1 / torch.sqrt(running_var + eps)
        std_weight = (bn_gamma * std).repeat(in_planes*h*w, 1).t().contiguous().view(out_planes, in_planes, h, w)
        new_conv_weight = std_weight * conv_weight
        new_conv_bias = bn_beta - running_mean * std * bn_gamma

        self._bn = None
        self._conv = nn.Conv2d(in_channels=self._conv.in_channels, out_channels=self._conv.out_channels,
                              kernel_size=self._conv.kernel_size,
                              padding=self._conv.padding, stride=self._conv.stride, dilation=self._conv.dilation,
                              groups=self._conv.groups,
                              bias=True)
        self._conv.weight.data = new_conv_weight
        self._conv.bias.data = new_conv_bias
