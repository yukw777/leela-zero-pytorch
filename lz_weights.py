import argparse
import torch

from yureka_go import Network

from flambe.compile import load_state_from_file


parser = argparse.ArgumentParser()
parser.add_argument('residual_channels', type=int)
parser.add_argument('residual_layers', type=int)
parser.add_argument('state_dir')
parser.add_argument('out_weight')
args = parser.parse_args()

n = Network(19, 18, args.residual_channels, args.residual_layers)
n.load_state_dict(load_state_from_file(args.state_dir, map_location=torch.device('cpu')))
n.to_leela_weights(args.out_weight)
