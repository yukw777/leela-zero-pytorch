# Leela Zero Pytorch
![](https://github.com/yukw777/leela-zero-pytorch/workflows/Test/badge.svg)

Generating Leela Zero weights via supervised training with PyTorch

# Dataset Generation
I used the KGS game records: https://u-go.net/gamerecords/

Leela zero comes with a tool that turns sgf files into datasets that can be fed into neural networks. See https://github.com/leela-zero/leela-zero#supervised-learning

# Training
Put the generated data into the directories specified in the flambe config files, then simply run the training pipeline via flambe.

Once the trainingF is over, you can use the `lz_weights.py` script to turn the PyTorch/Flambe weights into Leela Zero weights.

# Pretrained Weights
Please see [weights](weights) folder for pretrained weights.
