import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A convolutional block with a convolution layer, batchnorm and an optional relu
    """

    def __init__(self, in_channels, out_channels, kernel_size, relu=True):
        super(ConvBlock, self).__init__()
        # we only support the kernel sizes of 1 and 3
        assert kernel_size in (1, 3)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=1 if kernel_size == 3 else 0,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.relu = nn.ReLU(inplace=True) if relu else None

        # initializations
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x += self.beta.view(1, self.bn.num_features, 1, 1).expand_as(x)
        return self.relu(x) if self.relu is not None else x


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 3)
        self.conv2 = ConvBlock(out_channels, out_channels, 3, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += identity
        out = self.relu(out)

        return out


class Network(nn.Module):

    def __init__(self, board_size, in_channels, residual_channels, residual_layers):
        super(Network, self).__init__()
        self.conv_input = ConvBlock(in_channels, residual_channels, 3)
        self.residual_tower = nn.Sequential(
            *[ResBlock(residual_channels, residual_channels) for _ in range(residual_layers)])
        self.policy_conv = ConvBlock(residual_channels, 2, 1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)
        self.value_conv = ConvBlock(residual_channels, 1, 1)
        self.value_fc = nn.Sequential(
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # first conv layer
        x = self.conv_input(x)

        # residual tower
        x = self.residual_tower(x)

        # policy head
        pol = self.policy_conv(x)
        pol = self.policy_fc(torch.flatten(pol, start_dim=1))

        # value head
        val = self.value_conv(x)
        val = self.value_fc(torch.flatten(val, start_dim=1))

        return pol, val
