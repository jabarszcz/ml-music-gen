import librosa
import matplotlib.pyplot as plt
import numpy as np

def plot(stfted, filename=None, show=False):
    plt.clf()
    librosa.display.specshow(
        librosa.logamplitude(np.abs(stfted)**2, ref_power=np.max),
        y_axis='log', x_axis='time'
    )
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()

