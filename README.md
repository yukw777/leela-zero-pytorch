# Leela Zero Pytorch
![](https://github.com/yukw777/leela-zero-pytorch/workflows/Test/badge.svg)

Generating Leela Zero weights via supervised training with PyTorch

# Dataset Generation
I used the KGS game records: https://u-go.net/gamerecords/

Leela Zero comes with a tool that turns sgf files into datasets that can be fed into neural networks. See https://github.com/leela-zero/leela-zero#supervised-learning for more details.

1. Download desired sgf files.
1. Concatenate them into one file so that you can feed it into Leela Zero, e.g. `cat *.sgf > train.sgf`.
1. Feed the concatenated sgf file to Leela Zero, and it will generate a bunch of gzipped files. Divide them appropriate into train, validation and test folders. The default locations are `data/train`, `data/val` and `data/test` from the root of this repo.

# Training
1. Clone the repo and cd into the directory.
1. Install `leela-zero-pytorch`, by running `pip install .` or `pip install -e .` (editable mode).
1. Run `lzp-train`. Example: `lzp-train network=small` or `lzp-train network=huge dataset.train_dir=some/dataset/train train.gpus=1`.
1. Once the training is over, run `lzp-weights`. Example `lzp-weights path/to/checkpoint.ckpt weights.txt`.

# Pretrained Weights
Please see [weights](weights) folder for pretrained weights. You can use them directly with Leela Zero.
