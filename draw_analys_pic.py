import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import os, scipy

shred = 256
train_r = 5
data_folder_path = 'audioBook'
txt_folder = 'All_Slices_lab'
wav_folder = 'All_Slices_wav_24k'
outfile = 'data_audioBook.npz'
prefix_folder = 'F:\\big_data'
n_fft = 1024
win_length = int(50 * 16)
hop_length = int(12.5 * 16)
styles_kind = 10
style_dim = 2


wav_path = '1.wav'
# print(wav_path)
sr, data = scipy.io.wavfile.read(wav_path)
stftm_matrix = librosa.core.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

print(stftm_matrix)

plt.figure(figsize=(12, 8))
D = librosa.amplitude_to_db(stftm_matrix, ref=np.max)
plt.subplot(2, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.show()