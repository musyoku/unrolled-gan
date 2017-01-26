# -*- coding: utf-8 -*-
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu-device", type=int, default=0)
parser.add_argument("--image-dir", type=str, default="images")
parser.add_argument("--model-dir", type=str, default="model")
parser.add_argument("--plot-dir", type=str, default="plot")
parser.add_argument("-k", "--unrolling-steps", type=int, default=10)

# seed
parser.add_argument("--seed", type=int, default=None)

args = parser.parse_args()