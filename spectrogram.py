import librosa
import numpy as np

import matplotlib.pyplot as plt

def plot(stft, data, filename=None, show=False, diff=False):
    plt.clf()
    ref = 0 if diff else np.max
    librosa.display.specshow(
        librosa.logamplitude(np.abs(data)**2, ref_power=ref),
        y_axis='log', x_axis='time',
        sr=stft.samplerate, hop_length=stft.hop_len
    )
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()
