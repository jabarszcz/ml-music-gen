import random

import librosa
import torch
import numpy as np
from torch.utils.data import *

import spectrogram

class STFT:
    """A simple class to encapsulate a signal and its STFT transformation"""
    REPR_STRINGS = ["phase", "delta", "complex"]
    MODE_PHASE=0
    MODE_DELTA=1
    MODE_COMPLEX=2

    MODE2_MAG=0
    MODE2_POWER=1
    MODE2_DB=2

    def __init__(self, filter_len=2048, hop_len=None,
                 signal=None, filename=None, samplerate=None, mode=None, mode2=None):
        self.filter_len = filter_len
        self.hop_len = hop_len
        self.samplerate = samplerate
        self.mode = mode if mode is not None else STFT.MODE_DELTA
        self.mode2 = mode2 if mode2 is not None else STFT.MODE2_MAG
        if signal is not None:
            self.set_signal(signal)
        elif filename is not None:
            self.load(filename)

    def set_signal(self, signal):
        self.signal = signal
        self.stfted = self.stft(signal)
        self.real = self.stft_to_real(self.stfted)

    def load(self, filename):
        self.filename = filename
        signal, self.samplerate = librosa.load(filename, sr=self.samplerate)
        self.set_signal(signal)

    def save(self, filename, data=None):
        real = data if data is not None else self.real
        signal = self.istft(self.real_to_stft(real))
        librosa.output.write_wav(filename, signal, self.samplerate)

    def get_freqs(self):
        return self.real[:,:,0]

    def get_phases(self):
        return self.real[:,:,1]

    def get_tensor(self):
        return self.real_to_tensor(self.real)

    def stft(self, signal=None):
        return librosa.stft(
            signal if signal is not None else self.signal,
            n_fft=self.filter_len,
            hop_length=self.hop_len
        )

    def istft(self, stfted=None):
        return librosa.istft(
            stfted if stfted is not None else self.stfted,
            win_length=self.filter_len,
            hop_length=self.hop_len
        )

    def plot(self, data=None, filename=None, show=False, difference=False):
        real = data if data is not None else self.real
        stfted = self.real_to_stft(real)
        if data is not None and difference:
            stfted = self.stfted - stfted
        else:
            difference = False
        spectrogram.plot(stfted, filename=filename, show=show, diff=difference)

    def get_loss(self, data=None):
        """Calculate the MSE on the original audio wave"""
        real = data if data is not None else self.real
        stfted = self.real_to_stft(real)
        signal = self.istft(stfted)
        return np.mean(
            (np.resize(self.signal, signal.shape) - signal) ** 2
        )

    def real_to_tensor(self, real):
        return torch.from_numpy(np.ascontiguousarray(real.swapaxes(1,2)))

    def tensor_to_real(self, tensor):
        return tensor.numpy().swapaxes(1,2)

    def stft_to_real(self, stft):
        if self.mode == STFT.MODE_PHASE:
            real = self.stft_to_polar(stft)
        elif self.mode == STFT.MODE_DELTA:
            real = self.to_deltas(self.stft_to_polar(stft))
        else:
            real = self.stft_to_rect(stft)
        if self.mode2 == STFT.MODE2_DB:
            return self.power_to_db(self.mag_to_power(real))
        elif self.mode2 == STFT.MODE2_POWER:
            return self.mag_to_power(real)

    def real_to_stft(self, real):
        if self.mode2 == STFT.MODE2_DB:
            real = self.power_to_mag(self.db_to_power(real))
        elif self.mode2 == STFT.MODE2_POWER:
            real = self.power_to_mag(real)
        if self.mode == STFT.MODE_PHASE:
            return self.polar_to_stft(real)
        elif self.mode == STFT.MODE_DELTA:
            return self.polar_to_stft(self.from_deltas(real))
        else:
            return self.rect_to_stft(real)

    @staticmethod
    def mag_to_power(real):
        return np.stack([real[:,:,0]**2, real[:,:,1]], axis=2)

    @staticmethod
    def power_to_mag(real):
        return np.stack([real[:,:,0]**0.5, real[:,:,1]], axis=2)

    @staticmethod
    def power_to_db(real):
        return np.stack([np.log(np.maximum(1e-10,real[:,:,0])), real[:,:,1]],
                        axis=2)

    @staticmethod
    def db_to_power(real):
        return np.stack([np.exp(real[:,:,0]), real[:,:,1]], axis=2)

    @staticmethod
    def to_deltas(real):
        real = real.copy()
        phases = real[:,:,1]
        real[:,1:,1] = np.diff(phases, axis=1)
        return real

    @staticmethod
    def from_deltas(real):
        real = real.copy()
        phases = real[:,:,1]
        real[:,:,1] = np.cumsum(phases, axis=1)
        return real

    @staticmethod
    def stft_to_polar(stfted):
        stfted = stfted.T
        return np.stack([np.abs(stfted), np.angle(stfted)], axis=2)

    @staticmethod
    def polar_to_stft(real):
        return (real[:,:,0] * np.exp(1j*real[:,:,1])).T

    @staticmethod
    def stft_to_rect(stfted):
        c_ordered = np.ascontiguousarray(stfted.T)
        return c_ordered.view(np.float32).reshape(c_ordered.shape + (2,))

    @staticmethod
    def rect_to_stft(real):
        c_ordered = np.ascontiguousarray(real)
        return c_ordered.reshape((c_ordered.shape[0], -1)).view(np.complex64).T


class STFTDataset(Dataset):
    """A dataset class to load the STFT'ed samples"""

    def __init__(self, stft):
        self.stft = stft
        self.tensor = stft.get_tensor()

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, idx):
        return self.tensor[idx]


def concat_stft_dataset(inputs, **kwargs):
    sets = [STFTDataset(STFT(filename=i, **kwargs)) for i in inputs]
    return ConcatDataset(sets)

def split_random_samplers(size, *proportions):
    '''Make samplers that contain random subsets of given proportions

    `proportions` is a list of ratios in [0,1[. The last proportion is
    inferred.

    '''
    indices = range(size)
    random.shuffle(indices)
    lastidx = 0
    for p in proportions:
        newidx = lastidx + int(p * size)
        yield sampler.SubsetRandomSampler(indices[lastidx:newidx])
        lastidx = newidx
    yield sampler.SubsetRandomSampler(indices[lastidx:])
