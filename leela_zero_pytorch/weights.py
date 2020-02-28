import argparse

from leela_zero_pytorch.network import NetworkLightningModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('out_weight')
    args = parser.parse_args()

    m = NetworkLightningModule.load_from_checkpoint(args.checkpoint, map_location='cpu')
    m.model.to_leela_weights(args.out_weight)


if __name__ == '__main__':
    main()
