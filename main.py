from __future__ import print_function
import argparse, sys

import torch

from musicdata import *

parser = argparse.ArgumentParser(
    description="An experimental attempt at music"
    " generation with deep learning techniques"
)
parser.add_argument(
    "--disable-cuda", action="store_true", help="Disable CUDA"
)
parser.add_argument(
    "-i", "--inputs", action="append", help="Input files"
)
parser.add_argument(
    "--filter_length", type=int, default=2048,
    help="Length of the overlapping window for FFT"
)
parser.add_argument(
    "--hop_length", type=int, default=None,
    help="Stride of the overlapping window for FFT"
)

def main():
    args = parser.parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    if args.filter_length < args.hop_length:
        parser.error("Filter length must be greater than hop length",
                     file=sys.stderr)

    dataset = make_music_set(args.inputs, args.filter_length, args.hop_length)

if __name__ == "__main__":
    main()
