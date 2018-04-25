from __future__ import print_function
import argparse, itertools, logging

from subcommands import *
from musicdata import *

# LOGGING
logger = logging.getLogger()
logger.setLevel(logging.INFO)
myhandler = logging.StreamHandler()  # writes to stderr
myformatter = logging.Formatter(fmt='%(levelname)s: %(message)s')
myhandler.setFormatter(myformatter)
logger.addHandler(myhandler)

# PARSING
# parser with common arguments for inputs
parser_inputs = argparse.ArgumentParser(add_help=False)
parser_inputs.add_argument(
    "-i", "--inputs", action="append", nargs='+', default=[], help="Input files"
)
parser_inputs.add_argument(
    "--filter_length", type=int, default=2048,
    help="Length of the overlapping window for FFT"
)
parser_inputs.add_argument(
    "--hop_length", type=int, default=None,
    help="Stride of the overlapping window for FFT"
)
parser_inputs.add_argument(
    "-r", "--representation", default="delta",
    choices=STFT.REPR_STRINGS,
    help="Representation of the stft used for learning"
)

# parser with common pytorch arguments
parser_pytorch = argparse.ArgumentParser(add_help=False)
parser_pytorch.add_argument(
    "--disable-cuda", action="store_true", help="Disable CUDA"
)
parser_pytorch.add_argument(
    "-b", "--batch_size", type=int, default=128,
    help="The batch size used for training or running the models"
)

# Top-level parser
parser = argparse.ArgumentParser(
    description="An experimental attempt at music"
    " generation with deep learning techniques"
)
subparsers = parser.add_subparsers()

# train sub-command
parser_train = subparsers.add_parser(
    "train",
    description="Train a new VAE model",
    parents=[parser_inputs, parser_pytorch]
)
parser_train.set_defaults(func=train)
parser_train.add_argument(
    "-e", "--epochs", type=int, default=50,
    help="Maximum number of epochs for a run"
)
parser_train.add_argument(
    "-m", "--model", default="model.pb", help="filename of the saved model"
)
parser_train.add_argument(
    "--lr", default=0.015, type=float, help="Learning rate"
)
parser_train.add_argument(
    "-s", "--early-stopping", action="store_true", help="Activate early stopping"
)
parser_train.add_argument(
    "-p", "--patience", default=10, type=int, help="Patience for early stopping"
)

# run sub-command
parser_run = subparsers.add_parser(
    "run",
    description="Load and run an existing VAE model",
    parents=[parser_inputs, parser_pytorch]
)
parser_run.set_defaults(func=run)
parser_run.add_argument(
    "model", help="filename of the loaded model"
)

# experiments parser
parser_experiments = subparsers.add_parser(
    "experiments",
    description="Run some experiments on the data",
    parents=[parser_inputs]
)
parser_experiments.set_defaults(func=experiments)

def main():
    args = parser.parse_args()
    if 'inputs' in args:
        args.inputs = list(itertools.chain(*args.inputs))
        if args.filter_length < args.hop_length:
            parser.error("Filter length must be greater than hop length",
                         file=sys.stderr)

    if 'disable_cuda' in args:
        args.cuda = not args.disable_cuda and torch.cuda.is_available()

    args.func(args, parser)

if __name__ == "__main__":
    main()
