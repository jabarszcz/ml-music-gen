import librosa
import numpy as np

import matplotlib.pyplot as plt

def plot(stfted, filename=None, show=False, diff=False):
    plt.clf()
    ref = 0 if diff else np.max
    librosa.display.specshow(
        librosa.logamplitude(np.abs(stfted)**2, ref_power=ref),
        y_axis='log', x_axis='time'
    )
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()

