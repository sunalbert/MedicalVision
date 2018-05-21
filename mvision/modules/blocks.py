import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, is_bn=True,
                 is_relu=True):
        super(ConvBnRelu2D, self).__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self._bn = nn.BatchNorm2d(out_channels)
        self._relu = nn.ReLU(inplace=True)

        if self.bn is False:
            self.bn = None

        if is_relu is False:
            self.relu = None

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

        pass