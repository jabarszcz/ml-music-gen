from contextlib import contextmanager
from numpy import std
import copy
import csv
import logging

from musicdata import *

def add_noise(data, ratio):
    """Adds noise to the data, in proportion to its standart deviation"""
    data += np.random.normal(0, std(data)*ratio, data.shape)

def phases_zero(s):
    phases = s.get_phases()
    phases *= 0

def phases_noise(ratio):
    def fun(s):
        phases = s.get_phases()
        add_noise(phases, ratio)
    return fun

def freqs_noise(ratio):
    def fun(s):
        phases = s.get_freqs()
        add_noise(phases, ratio)
    return fun

experiments = [
    ("baseline", lambda s : s),
    ("phases_zero", phases_zero),
    ("phases_small_noise", phases_noise(0.1)),
    ("phases_big_noise", phases_noise(0.5)),
    ("freqs_small_noise", freqs_noise(0.1)),
    ("freqs_big_noise", freqs_noise(0.5)),
]

@contextmanager
def experiment_ctx(stft, name, losses_csv):
    stft = copy.deepcopy(stft)
    yield stft
    wavfile = "out_%s.wav" % name
    stft.save(wavfile)
    logging.info("Saved experiment %s to %s" % (name, wavfile))
    spectrogramfile = "spectrogram_%s.png" % name
    stft.plot(spectrogramfile)
    logging.info("Saved spectrogram %s to %s" % (name, spectrogramfile))
    loss = stft.get_loss()
    losses_csv.writerow({'name':name, 'losses':loss})
    logging.info("Added loss of %g for %s to csv file" % (loss, name))


def run_experiments(stft):
    with open('experiment_losses.csv', 'wb') as csvfile:
        fieldnames = ['name', 'losses']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for name, function in experiments:
            with experiment_ctx(stft, name, writer) as s:
                function(s)
