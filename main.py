from __future__ import print_function
import argparse, sys

import torch
from torch.utils.data import DataLoader

import librosa

from experiments import *
from musicdata import *

# LOGGING
logger = logging.getLogger()
logger.setLevel(logging.INFO)
myhandler = logging.StreamHandler()  # writes to stderr
myformatter = logging.Formatter(fmt='%(levelname)s: %(message)s')
myhandler.setFormatter(myformatter)
logger.addHandler(myhandler)

# PARSING
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
parser.add_argument(
    "--experiments", action="store_true", help="Run data experiments"
)

def main():
    args = parser.parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    if args.filter_length < args.hop_length:
        parser.error("Filter length must be greater than hop length",
                     file=sys.stderr)

    if args.experiments:
        s = STFT(args.filter_length, args.hop_length, filename=args.inputs[0], deltas=False)
        run_experiments(s)

if __name__ == "__main__":
    main()
