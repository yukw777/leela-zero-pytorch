import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Dict, Tuple, Any

from leela_zero_pytorch.dataset import DataPoint


class ConvBlock(nn.Module):
    """
    A convolutional block with a convolution layer, batchnorm (with beta) and
    an optional relu

    Note on the bias for the convolutional layer:
    Leela Zero actually uses the bias for the convolutional layer to represent
    the learnable parameters (gamma and beta) of the following batch norm layer.
    This was done so that the format of the weights file, which only has one line
    for the layer weights and another for the bias, didn't have to change when
    batch norm layers were added.

    Currently, Leela Zero only uses the beta term of batch norm, and sets gamma to 1.
    Then, how do you actually use the convolutional bias to produce the same results
    as applying the learnable parameters in batch norm? Let's first take
    a look at the equation for batch norm:

    y = gamma * (x - mean)/sqrt(var - eps) + beta

    Since Leela Zero sets gamma to 1, the equation becomes:

    y = (x - mean)/sqrt(var - eps) + beta

    Now, let `x_conv` be the output of a convolutional layer without the bias.
    Then, we want to add some bias to `x_conv`, so that when you run it through
    batch norm without `beta`, the result is the same as running `x_conv`
    through the batch norm equation with only beta mentioned above. In an equation form:

    (x_conv + bias - mean)/sqrt(var - eps) = (x_conv - mean)/sqrt(var - eps) + beta
    x_conv + bias - mean = x_conv - mean + beta * sqrt(var - eps)
    bias = beta * sqrt(var - eps)

    So if we set the convolutional bias to `beta * sqrt(var - eps)`, we get the desired
    output, and this is what LeelaZero does.

    In Tensorflow, you can tell the batch norm layer to ignore just the gamma term
    by calling `tf.layers.batch_normalization(scale=False)` and be done with it.
    Unfortunately, in PyTorch you can't set batch normalization layers to ignore only
    `gamma`; you can only ignore both `gamma` and `beta` by setting the affine
    parameter to False: `BatchNorm2d(out_channels, affine=False)`. So, ConvBlock sets
    batch normalization to ignore both, then simply adds a tensor after, which
    represents `beta`.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, relu: bool = True
    ):
        super().__init__()
        # we only support the kernel sizes of 1 and 3
        assert kernel_size in (1, 3)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=1 if kernel_size == 3 else 0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.beta = nn.Parameter(torch.zeros(out_channels))  # type: ignore
        self.relu = relu

        # initializations
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x += self.beta.view(1, self.bn.num_features, 1, 1).expand_as(x)
        return F.relu(x, inplace=True) if self.relu else x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 3)
        self.conv2 = ConvBlock(out_channels, out_channels, 3, relu=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += identity
        return F.relu(out, inplace=True)


class Network(nn.Module):
    def __init__(
        self,
        board_size: int,
        in_channels: int,
        residual_channels: int,
        residual_layers: int,
    ):
        super().__init__()
        self.conv_input = ConvBlock(in_channels, residual_channels, 3)
        self.residual_tower = nn.Sequential(
            *[
                ResBlock(residual_channels, residual_channels)
                for _ in range(residual_layers)
            ]
        )
        self.policy_conv = ConvBlock(residual_channels, 2, 1)
        self.policy_fc = nn.Linear(
            2 * board_size * board_size, board_size * board_size + 1
        )
        self.value_conv = ConvBlock(residual_channels, 1, 1)
        self.value_fc_1 = nn.Linear(board_size * board_size, 256)
        self.value_fc_2 = nn.Linear(256, 1)

    def forward(self, planes, target_pol, target_val):
        # first conv layer
        x = self.conv_input(planes)

        # residual tower
        x = self.residual_tower(x)

        # policy head
        pol = self.policy_conv(x)
        pol = self.policy_fc(torch.flatten(pol, start_dim=1))

        # value head
        val = self.value_conv(x)
        val = F.relu(self.value_fc_1(torch.flatten(val, start_dim=1)), inplace=True)
        val = torch.tanh(self.value_fc_2(val))

        return (pol, val), (target_pol, target_val)

    def to_leela_weights(self, filename: str):
        """
        Save the current weights in the Leela Zero format to the given file name.

        The Leela Zero weights format:

        The residual tower is first, followed by the policy head, and then the value
        head. All convolution filters are 3x3 except for the ones at the start of
        the policy and value head, which are 1x1 (as in the paper).

        Convolutional layers have 2 weight rows:
            1. convolution weights w/ shape [output, input, filter_size, filter_size]
            2. channel biases
        Batchnorm layers have 2 weight rows:
            1. batchnorm means
            2. batchnorm variances
        Innerproduct (fully connected) layers have 2 weight rows:
            1. layer weights w/ shape [output, input]
            2. output biases

        Therefore, the equation for the number of layers is

        n_layers = 1 (version number) +
                   2 (input convolution) +
                   2 (input batch norm) +
                   n_res (number of residual blocks) *
                   8 (first conv + first batch norm + second conv + second batch norm) +
                   2 (policy head convolution) +
                   2 (policy head batch norm) +
                   2 (policy head linear) +
                   2 (value head convolution) +
                   2 (value head batch norm) +
                   2 (value head first linear) +
                   2 (value head second linear)
        """
        with open(filename, "w") as f:
            # version tag
            f.write("1\n")
            for child in self.children():
                # newline unless last line (single bias)
                if isinstance(child, ConvBlock):
                    f.write(self.conv_block_to_leela_weights(child))
                elif isinstance(child, nn.Linear):
                    f.write(self.tensor_to_leela_weights(child.weight))
                    f.write(self.tensor_to_leela_weights(child.bias))
                elif isinstance(child, nn.Sequential):
                    # residual tower
                    for grand_child in child.children():
                        if isinstance(grand_child, ResBlock):
                            f.write(self.conv_block_to_leela_weights(grand_child.conv1))
                            f.write(self.conv_block_to_leela_weights(grand_child.conv2))
                        else:
                            err = (
                                "Sequential should only have ResBlocks, but found "
                                + str(type(grand_child))
                            )
                            raise ValueError(err)
                else:
                    raise ValueError("Unknown layer type" + str(type(child)))

    @staticmethod
    def conv_block_to_leela_weights(conv_block: ConvBlock) -> str:
        weights = []
        weights.append(Network.tensor_to_leela_weights(conv_block.conv.weight))
        # calculate beta * sqrt(var - eps)
        bias = conv_block.beta * torch.sqrt(
            conv_block.bn.running_var - conv_block.bn.eps  # type: ignore
        )
        weights.append(Network.tensor_to_leela_weights(bias))
        weights.append(
            Network.tensor_to_leela_weights(conv_block.bn.running_mean)  # type: ignore
        )
        weights.append(
            Network.tensor_to_leela_weights(conv_block.bn.running_var)  # type: ignore
        )
        return "".join(weights)

    @staticmethod
    def tensor_to_leela_weights(t: torch.Tensor) -> str:
        return " ".join([str(w) for w in t.detach().numpy().ravel()]) + "\n"


class NetworkLightningModule(pl.LightningModule):
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()
        self.hparams = hparams
        self.model = Network(
            hparams["board_size"],
            hparams["in_channels"],
            hparams["residual_channels"],
            hparams["residual_layers"],
        )

    def forward(self, planes, target_pol, target_val):
        return self.model(planes, target_pol, target_val)

    def loss(
        self,
        pred: Tuple[torch.Tensor, torch.Tensor],
        target: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_move, pred_val = pred
        target_move, target_val = target
        cross_entropy_loss = F.cross_entropy(pred_move, target_move)
        mse_loss = F.mse_loss(pred_val.squeeze(), target_val)
        return mse_loss, cross_entropy_loss, mse_loss + cross_entropy_loss

    def training_step(self, batch: DataPoint, batch_idx: int) -> Dict:
        pred, target = self.forward(*batch)
        mse_loss, cross_entropy_loss, loss = self.loss(pred, target)
        return {
            "loss": loss,
            "log": {
                "training_loss": loss,
                "training_mse_loss": mse_loss,
                "training_ce_loss": cross_entropy_loss,
            },
        }

    def validation_step(self, batch: DataPoint, batch_idx: int) -> Dict:
        pred, target = self.forward(*batch)
        mse_loss, cross_entropy_loss, loss = self.loss(pred, target)
        return {
            "val_loss": loss,
            "val_mse_loss": mse_loss,
            "val_ce_loss": cross_entropy_loss,
        }

    def validation_epoch_end(self, outputs):
        loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        mse_loss_mean = torch.stack([x["val_mse_loss"] for x in outputs]).mean()
        ce_loss_mean = torch.stack([x["val_ce_loss"] for x in outputs]).mean()
        return {
            "log": {
                "validation_loss": loss_mean,
                "validation_mse_loss": mse_loss_mean,
                "validation_ce_loss": ce_loss_mean,
            },
            "val_loss": loss_mean,
        }

    def test_step(self, batch: DataPoint, batch_idx: int) -> Dict:
        pred, target = self.forward(*batch)
        mse_loss, cross_entropy_loss, loss = self.loss(pred, target)
        return {
            "test_loss": loss,
            "test_mse_loss": mse_loss,
            "test_ce_loss": cross_entropy_loss,
        }

    def test_epoch_end(self, outputs):
        loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()
        mse_loss_mean = torch.stack([x["test_mse_loss"] for x in outputs]).mean()
        ce_loss_mean = torch.stack([x["test_ce_loss"] for x in outputs]).mean()
        return {
            "log": {
                "test_loss": loss_mean,
                "test_mse_loss": mse_loss_mean,
                "test_ce_loss": ce_loss_mean,
            },
            "val_loss": loss_mean,
        }

    def configure_optimizers(self):
        # taken from leela zero
        # https://github.com/leela-zero/leela-zero/blob/db5569ce8d202f77154f288c21d3f2fa228f9aa3/training/tf/tfprocess.py#L190-L191
        sgd_opt = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            momentum=0.9,
            nesterov=True,
            weight_decay=1e-4,
        )
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            sgd_opt, verbose=True, min_lr=5e-6
        )
        return [sgd_opt], [lr_sched]
