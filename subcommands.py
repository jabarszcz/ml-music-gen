from __future__ import print_function
import sys

import torch
from torch.utils.data import DataLoader

from experiments import *
from musicdata import *
from vae import *

def train(args, parser):
    dataset = make_dataset(args, parser)
    train_sampler, valid_sampler = split_random_samplers(len(dataset), 0.7)
    train_loader = DataLoader(dataset, args.batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, args.batch_size, sampler=valid_sampler)

    vae = VAE(
        args.filter_length/2 + 1,
        enable_cuda=args.cuda,
        filters=32,
        z_dim=512
    )

    with open('learning.csv', 'wb') as csvfile:
        def write_data(data_dict):
            if write_data.writer is None:
                fieldnames = data_dict.keys()
                write_data.writer = csv.DictWriter(
                    csvfile,
                    fieldnames=fieldnames
                )
                write_data.writer.writeheader()

            write_data.writer.writerow(data_dict)
        write_data.writer = None

        train_vae(
            vae,
            train_loader,
            valid_loader,
            args.epochs,
            0.004,
            results_cb=write_data
        )

    vae.save(args.model)

def run(args, parser):
    dataset = make_dataset(args, parser)
    vae = VAE.load(args.model)

    for song_dataset in dataset.datasets:
        res, reconst_loss, kl_loss = run_vae(
            vae,
            DataLoader(song_dataset, args.batch_size),
            keep_results=True
        )
        stft = song_dataset.stft
        newstfted = stft.tensor_to_real(res.data)
        print("total loss:", reconst_loss + kl_loss)
        print("reconstruction loss:", reconst_loss)
        print("KL loss:", kl_loss)
        print("signal loss:", stft.get_loss(newstfted))
        stft.plot(newstfted, filename="result.png")
        stft.save("result.wav", data=newstfted)

def experiments(args, parser):
    s = STFT(
        args.filter_length,
        args.hop_length,
        filename=args.inputs[0],
        deltas=False
    )
    run_experiments(s)

def make_dataset(args, parser):
    if not len(args.inputs) > 0:
        parser.error("There should be at least an input song to run the model",
                     file=sys.stderr)
    return concat_stft_dataset(args.inputs, args.filter_length, args.hop_length)
