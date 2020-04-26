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
1. Run `lzp-train`. Example: `lzp-train network=small` or `lzp-train network=huge train.dataset.dir_path=some/dataset/train pl_trainer.gpus=-1`.
1. Once the training is over, run `lzp-weights`. Example `lzp-weights path/to/checkpoint.ckpt weights.txt`.

# Pretrained Weights
I have trained three networks using the same training data of about 1.3 million positions generated from the KGS game records. The huge network ([weights](weights/leela-zero-pytorch-huge.txt), [config](leela_zero_pytorch/conf/network/huge.yaml)) is strongest, followed by the big network ([weights](weights/leela-zero-pytorch-bg.txt), [config](leela_zero_pytorch/conf/network/big.yaml)), and followed by the small network ([weights](weights/leela-zero-pytorch-sm.txt), [config](leela_zero_pytorch/conf/network/small.yaml)). You can use them directly with Leela Zero.
