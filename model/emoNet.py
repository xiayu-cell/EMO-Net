import time
from torch.autograd import Variable
from torch import nn, Tensor
from torch.nn import functional as F
import torch
import math

from torchvision import models

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class EfficientChannelAttention(nn.Module):  # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        # calculate the kernel size of 1D conv
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(x)
        #after GAP,Conv1D,Sigmod, we get the weight of each channel containing cross-channel information
        return out


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        # self.use_shortcut = stride == 1 and in_channel == out_channel
        if stride == 1 and in_channel == out_channel:
            self.dowmSample = None
        else:
            self.dowmSample = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channel)
            )
        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            # No non-linear activation function is used when outputting from the inverted residual structure
        ])
        self.conv = nn.Sequential(*layers)
        self.channel = EfficientChannelAttention(out_channel)

    def forward(self, x):
        out = self.conv(x)
        ECA_out = self.channel(out)
        #local attention to the main branch
        out = out * ECA_out
        #fusion the main branch and auxiliary branch
        if self.dowmSample == None:
            return 0.8*x + out
        else:
            return 0.8*self.dowmSample(x)+out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=7, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s

            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def speed(model, name):
    t0 = time.time()
    input = torch.rand(1, 3, 224, 224).cuda()  # input = torch.rand(1,3,224,224).cuda()
    input = Variable(input, volatile=True)
    t1 = time.time()

    model(input)

    t2 = time.time()
    for i in range(10):
        model(input)
    t3 = time.time()

    torch.save(model.state_dict(), "test_%s.pth" % name)
    print('%10s : %f' % (name, t3 - t2))

# if __name__ == '__main__':
#     net = MobileNetV2()
#     input = torch.rand(1, 3, 224, 224)
#     output = net(input)
#     total = sum(p.numel() for p in net.parameters())
#     print("Total params: %.2fM" % (total / 1e6))
#
#     from thop import profile
#
#     input = torch.randn(1, 3, 224, 224)
#     flops, params = profile(net, inputs=(input,))
#     print('flops:{}G'.format(flops/1e9))
#     print('params:{}M'.format(params/1e6))

    #calculate the computational complexity of a model
    # input = torch.randn(2,3, 224, 224)
    # calc_flops(net,input)

    # print(output[0].shape).
