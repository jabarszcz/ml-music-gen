from __future__ import print_function
import argparse, sys

import torch
from torch.utils.data import DataLoader

import librosa

from experiments import *
from musicdata import *
from vae import *

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
    "-e", "--epochs", type=int, default=None,
    help="Maximum number of epochs for a run"
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

    if args.epochs:
        train(args)

def train(args):
    dataset = concat_stft_dataset(args.inputs, args.filter_length, args.hop_length)
    data_loader = DataLoader(dataset, 128, shuffle=True)

    vae = VAE(args.filter_length/2 + 1, enable_cuda=args.cuda, filters=16, z_dim=512)

    with open('learning.csv', 'wb') as csvfile:
        fieldnames = ['epoch', 'epochs', 'iter', 'iters',
                      'total_loss', 'reconst_loss', 'kl_divergence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        def write_data(data_dict):
            writer.writerow(data_dict)

        train_vae(vae, data_loader, args.epochs, 0.03, results_cb=write_data)

    vae.eval()
    song_dataset = dataset.datasets[0]
    res = eval_result(vae, DataLoader(song_dataset, 128))
    stft = song_dataset.stft

    newstfted = stft.tensor_to_real(res.data)
    print("loss:", stft.get_loss(newstfted))
    stft.plot(newstfted, filename="result.png")
    stft.save("result.wav", data=newstfted)

if __name__ == "__main__":
    main()
