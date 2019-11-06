import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


def conv1x1(in_channels, out_channels):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Network(nn.Module):

    def __init__(self, board_size, in_channels, residual_channels, residual_layers):
        super(Network, self).__init__()
        self.conv_input = conv3x3(in_channels, residual_channels)
        self.bn_input = nn.BatchNorm2d(residual_channels)
        self.relu_input = nn.ReLU(inplace=True)
        self.residual_tower = nn.Sequential(
            *[ResBlock(residual_channels, residual_channels) for _ in range(residual_layers)])
        self.policy_conv = conv1x1(residual_channels, 2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)
        self.value_conv = conv1x1(residual_channels, 1)
        self.value_fc = nn.Sequential(*[
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, ResBlock):
                # Zero-initialize the last BN in each residual branch,
                # so that the residual branch starts with zeros, and each residual block behaves like an identity.
                # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        # first conv layer
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = self.relu_input(x)

        # residual tower
        x = self.residual_tower(x)

        # policy head
        pol = self.policy_conv(x)
        pol = self.policy_fc(torch.flatten(pol, start_dim=1))

        # value head
        val = self.value_conv(x)
        val = self.value_fc(torch.flatten(val, start_dim=1))

        return pol, val
