import librosa
import numpy as np
from torch.utils.data import *

import spectrogram

class MusicDataset(Dataset):
    """A dataset class to load the STFT'ed samples"""

    def __init__(self, input_file, filter_len=2048, hop_len=None):
        self.filter_len, self.hop_len = filter_len, hop_len
        self.samples, self.rate = librosa.load(input_file, sr=None)
        self.stfted = librosa.stft(
            self.samples,
            n_fft=filter_len,
            hop_length=hop_len
        )
        c_ordered = np.ascontiguousarray(self.stfted.T)
        self.data = c_ordered.view(np.float32).reshape(c_ordered.shape + (2,))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        self.data[idx]

    def plot(self, **kwargs):
        spectrogram.plot(self.stfted, **kwargs)

    def inverse(self, data=None):
        """Run the inverse of the stft to get back the audio wave"""
        return librosa.istft(
            data if data else self.stfted,
            win_length=self.filter_len,
            hop_length=self.hop_len
        )

    def get_loss(self, data=None):
        """Calculate the MSE on the original audio wave"""
        inv = self.inverse(data)
        return np.mean((np.resize(self.samples, inv.shape) - inv) ** 2)

def make_music_set(inputs, filter_len, hop_len):
    music_sets = [MusicDataset(i, filter_len, hop_len) for i in inputs]
    return ConcatDataset(music_sets)
