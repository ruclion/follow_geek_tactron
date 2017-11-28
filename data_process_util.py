import numpy as np
import os
from math import floor, ceil
import random
import librosa
import scipy.io.wavfile
import audio

data_folder_path = 'data'
wav_path = 'test.wav'
outfile = 'data.npz'
n_fft = 1024
win_length = int(50 * 16)
hop_length = int(12.5 * 16)
styles_kind = 10
style_dim = 2


data_inp = []
data_inp_mask = []
data_mel_gtruth = []
data_spec_gtruth = []
data_speaker = []
data_style = []

char_map = {}
cnt = 1
max_len = 0
wav_file_num = 0


data_folder_path = os.path.realpath(data_folder_path)

for root, sub_dirs, files in os.walk(data_folder_path):
    for wav_file in files:
        if wav_file[-4:] == '.wav':
            wav_file_num += 1
            wav_file = os.path.join(data_folder_path, wav_file)
            print('??:', wav_file)
            sr, data = scipy.io.wavfile.read(wav_file)
            stftm_matrix = np.abs(librosa.core.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
            S = librosa.feature.melspectrogram(data, sr=sr, n_fft=n_fft, hop_length=hop_length)
            stftm_matrix = stftm_matrix.T
            S = S.T
            data_spec_gtruth.append(stftm_matrix)
            data_mel_gtruth.append(S)

            txt_file = wav_file[0:-3] + 'txt'
            with open(txt_file, "r") as f:
                text = f.read()
            text_map = []
            max_len = max(max_len, len(text))
            data_inp_mask.append(len(text))
            print(text, len(text))
            for i in range(len(text)):
                if text[i] in char_map:
                    text_map.append(char_map[text[i]])
                else:
                    char_map[text[i]] = cnt
                    text_map.append(cnt)
                    cnt += 1
            data_inp.append(text_map)

            data_speaker.append(0)

            data_style.append(0.5 * np.ones((styles_kind, style_dim), dtype=np.float32))

for i in range(wav_file_num):
    for j in range(max_len - data_inp_mask[i]):
        data_inp[i].append(0)
    data_inp[i] = np.array(data_inp[i])

data_inp = np.array(data_inp)
data_inp_mask = np.array(data_inp_mask)
data_mel_gtruth = np.array(data_mel_gtruth)
data_spec_gtruth = np.array(data_spec_gtruth)
data_speaker = np.array(data_speaker)
data_style = np.array(data_style)
data_all_style = np.array(0.5 * np.ones((1, styles_kind, style_dim), dtype=np.float32))

print(data_inp)

np.savez(outfile, inp=data_inp, inp_mask=data_inp_mask, mel_gtruth=data_mel_gtruth, spec_gtruth=data_spec_gtruth,
         speaker=data_speaker, style=data_style, all_style = data_all_style)
