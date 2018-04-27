from __future__ import print_function
import logging, os, sys

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

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

        optimizer = torch.optim.Adam(vae.parameters(), args.lr)
        scheduler = LambdaLR(optimizer, [lambda epoch: 0.1 ** (epoch/100.0)])

        train_vae(
            vae,
            train_loader,
            valid_loader,
            args.epochs,
            results_cb=write_data,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=args.early_stopping,
            patience=args.patience
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
        metrics_dict = {
            'total_loss':reconst_loss + kl_loss,
            'reconst_loss':reconst_loss,
            'kl_loss':kl_loss,
            'signal_loss':stft.get_loss(newstfted)
        }

        # Save all the things
        dirname, filename = os.path.split(stft.filename)
        def mkpath(template):
            return os.path.join(dirname, template.format(filename))
        losses_path = mkpath('losses_{}.csv')
        spectrogram_path = mkpath("spectrogram_{}.png")
        out_path = mkpath("out_{}.wav")

        logging.info("Losses for {}: {}".format(filename, metrics_dict))
        logging.info("Saving losses for {} to {}".format(filename, losses_path))
        with open(losses_path, 'wb') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics_dict.keys())
            writer.writeheader()
            writer.writerow(metrics_dict)

        logging.info(
            "Saving spectrogram for {} to {}".format(filename, spectrogram_path)
        )
        stft.plot(newstfted, filename=spectrogram_path)

        logging.info(
            "Saving decoded version of {} to {}".format(filename, out_path)
        )
        stft.save(out_path, data=newstfted)

def experiments(args, parser):
    # Experiments are run on the first input file only
    dataset = make_dataset(args, parser)
    run_experiments(dataset.datasets[0].stft)

def make_dataset(args, parser):
    if not len(args.inputs) > 0:
        parser.error("There should be at least an input song to run the model",
                     file=sys.stderr)
    mode = STFT.REPR_STRINGS.index(args.representation)
    return concat_stft_dataset(
        args.inputs,
        filter_len=args.filter_length,
        hop_len=args.hop_length,
        mode=mode
    )
