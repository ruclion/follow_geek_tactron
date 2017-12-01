import numpy as np
import os
from math import floor, ceil
import random
import librosa
import scipy.io.wavfile
import audio


shred = 256
train_r = 5
data_folder_path = 'audioBook'
txt_folder = 'All_Slices_lab'
wav_folder = 'All_Slices_wav_24k'
outfile = 'data_audioBook.npz'
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
max_timestamp = 0
wav_file_num = 0
tot_nouse_num = 0


data_wav_folder_path = os.path.realpath(os.path.join(data_folder_path, wav_folder))
data_txt_folder_path = os.path.realpath(os.path.join(data_folder_path, txt_folder))

for root, sub_dirs, files in os.walk(data_wav_folder_path):
    # print(sub_dirs)
    for name_dir in sub_dirs:
        for single_wav_path in os.listdir(os.path.join(data_wav_folder_path, name_dir)):
            wav_file_num += 1
            wav_path = os.path.join(os.path.join(data_wav_folder_path, name_dir), single_wav_path)
            print(wav_path)
            sr, data = scipy.io.wavfile.read(wav_path)
            stftm_matrix = np.abs(librosa.core.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
            S = librosa.feature.melspectrogram(data, sr=sr, n_fft=n_fft, hop_length=hop_length)
            stftm_matrix = stftm_matrix.T
            S = S.T

            txt_file = os.path.join(data_txt_folder_path, name_dir, single_wav_path)
            txt_file = txt_file[0:-3] + 'lab'
            # print(txt_file)
            with open(txt_file, "r") as f:
                text = f.read()
            if len(text) > 64:
                print(text, len(text))
                wav_file_num -= 1
                tot_nouse_num += 1
                continue
            text_map = []
            max_len = max(max_len, len(text))
            print(type(stftm_matrix), stftm_matrix.shape)
            max_timestamp = max(max_timestamp, stftm_matrix.shape[0])
            data_spec_gtruth.append(stftm_matrix)
            data_mel_gtruth.append(S)
            data_inp_mask.append(len(text))

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
            if wav_file_num >= shred:
                break
        if wav_file_num >= shred:
            break
    break
'''    
    
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
'''
max_timestamp = (max_timestamp + 4) // 5 * 5
print('tot num:', wav_file_num, max_len, tot_nouse_num, max_timestamp)
for i in range(wav_file_num):
    # max_len - data_inp_mask[i]
    for j1 in range(max_len - data_inp_mask[i]):
        data_inp[i].append(0)
    tmp_mel_list = data_mel_gtruth[i].tolist()
    tmp_spec_list = data_spec_gtruth[i].tolist()
    times = max_timestamp - data_mel_gtruth[i].shape[0]
    for j2 in range(times):
        tmp_mel_list.append(np.zeros_like(data_mel_gtruth[i][0], dtype=np.float32))
    for j3 in range(times):
        tmp_spec_list.append(np.zeros_like(data_spec_gtruth[i][0], dtype=np.float32))
    data_mel_gtruth[i] = np.array(tmp_mel_list)
    data_spec_gtruth[i] = np.array(tmp_spec_list)
    print('now:', data_mel_gtruth[i].shape, data_spec_gtruth[i].shape)

    data_inp[i] = np.array(data_inp[i])

data_inp = np.array(data_inp)
data_inp_mask = np.array(data_inp_mask)
data_mel_gtruth = np.array(data_mel_gtruth)
data_spec_gtruth = np.array(data_spec_gtruth)
data_speaker = np.array(data_speaker)
data_style = np.array(data_style)
data_all_style = np.array(0.5 * np.ones((styles_kind, style_dim), dtype=np.float32))


print('final:', data_inp.shape, data_inp.shape, data_mel_gtruth.shape, data_spec_gtruth.shape, data_all_style.shape)

# print(data_inp)

np.savez(outfile, inp=data_inp, inp_mask=data_inp_mask, mel_gtruth=data_mel_gtruth, spec_gtruth=data_spec_gtruth,
         speaker=data_speaker, style=data_style, all_style = data_all_style)
