from __future__ import print_function
from __future__ import division

import librosa
import numpy as np

n_fft = 1024
win_length = int(50 * 16)
hop_length = int(12.5 * 16)
alpha = 0.97

BITS = 16
MAX = 2 ** (BITS - 1)

def de_emphasis(wav_arr, alpha=0.97):
    ret = np.zeros_like(wav_arr)
    lens = len(wav_arr)
    ret[0] = wav_arr[0]
    for idx in range(1, lens):
        ret[idx] = wav_arr[idx] + alpha * ret[idx-1]
    return ret

def invert_spectrogram(spec, coef=1.2, out_fn=None, sr=16000, return_float=True, de_emp=True):
    spec = np.power(np.exp(spec.T), coef)
    wav_dots = (spec.shape[1] - 1) * hop_length
    y = np.random.uniform(low=-1, high=1, size=(wav_dots,))
    for i in range(50):
        s_m = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        s_m = spec * s_m / np.abs(s_m)
        y = librosa.core.istft(s_m, hop_length=hop_length, win_length=win_length)
    if de_emp:
        y = de_emphasis(y, 0.85)
    if return_float:
        y = y / np.max(np.abs(y))
    return y

def invert_spectrogram_noTandPower(spec, coef=1.2, out_fn=None, sr=16000, return_float=True, de_emp=True):
    # spec = np.power(np.exp(spec.T), coef)
    wav_dots = (spec.shape[1] - 1) * hop_length
    y = np.random.uniform(low=-1, high=1, size=(wav_dots,))
    for i in range(50):
        s_m = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        s_m = spec * s_m / np.abs(s_m)
        y = librosa.core.istft(s_m, hop_length=hop_length, win_length=win_length)
    if de_emp:
        y = de_emphasis(y, 0.85)
    if return_float:
        y = y / np.max(np.abs(y))
    return y
