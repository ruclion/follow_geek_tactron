import tensorflow as tf
import numpy as np
import librosa
import os
import scipy.io.wavfile
import audio



a = np.array([0, 1, 2, 3, 4])
b = np.array([2, 4])
c = a[0:3]
# print(c)

path = 'data.npz'
data = np.load(path)

inp = data['mel_gtruth']
print(inp)



'''
def griffin_lim(stftm_matrix, shape, min_iter=20, max_iter=50, delta=20):
    y = np.random.random(shape)
    y_iter = []

    for i in range(max_iter):
        if i >= min_iter and (i - min_iter) % delta == 0:
            y_iter.append((y, i))
        stft_matrix = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_matrix = stftm_matrix * stft_matrix / np.abs(stft_matrix)
        y = librosa.core.istft(stft_matrix, hop_length=hop_length, win_length=win_length)
    y_iter.append((y, max_iter))

    return y_iter



# assume 1 channel wav file

wav_path = 'test.wav'
n_fft = 1024
win_length = int(50 * 16)
hop_length = int(12.5 * 16)

print(os.path.realpath(wav_path))

# data, sr = librosa.load(wav_path)

# D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

sr, data = scipy.io.wavfile.read(wav_path)
print(np.shape(data))
print(min(data))
    # 由 STFT -> STFT magnitude
stftm_matrix = np.abs(librosa.core.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length))

print(np.shape(stftm_matrix))
# for i in range(len(stftm_matrix)):
#     print(max(stftm_matrix[i]), min(stftm_matrix[i]))
S = librosa.feature.melspectrogram(data, sr=sr, n_fft=n_fft, hop_length=hop_length)
for i in range(len(S)):
    print(max(S[i]), min(S[i]))
print(S)





    # + random 模拟 modification
stftm_matrix_modified = stftm_matrix + np.random.random(stftm_matrix.shape)

    # Griffin-Lim 估计音频信号
predict_data = audio.invert_spectrogram_noTandPower(stftm_matrix_modified)
# print(np.shape(predict_data), max(predict_data), min(predict_data))
# y_iters = griffin_lim(stftm_matrix_modified, data.shape)
# np_y = np.array(y_iters)
print(predict_data.shape)

scipy.io.wavfile.write('out.wav', sr, predict_data)

'''


# init_a = 3 * np.ones((2, 2), dtype=np.float32)
#
# a = tf.get_variable(name='a', dtype=tf.float32, initializer=init_a)
# # a = tf.constant(3, dtype=tf.float32, shape=(2, 2))
# # print(a.get_shape()[-1])
# b = tf.constant(5, dtype=tf.float32, shape=(2, 2))
#
# w = tf.get_variable('w', shape=(1), dtype=tf.float32, initializer=tf.constant_initializer(1.0))
# b_pre = w * a
# loss = tf.reduce_mean(tf.abs(b_pre - b))
#
# opt = tf.train.AdamOptimizer(0.1)
# grads_and_vars = opt.compute_gradients(loss)
#
# session = tf.Session()
#
#
#
#
#
# train_upd = opt.apply_gradients(grads_and_vars)
# session.run(tf.initialize_all_variables())
# print(session.run(a))
# print(session.run(grads_and_vars))
#
# session.run(train_upd)
#
# print(session.run(a))
# print(session.run(w))
#
# session.run(train_upd)
#
# print(session.run(a))
# print(session.run(w))
#
# session.run(train_upd)
#
# print(session.run(a))
# print(session.run(w))
#
# session.run(train_upd)
#
# print(session.run(a))
# print(session.run(w))
#
# session.run(train_upd)
#
# print(session.run(a))
# print(session.run(w))
#
# session.run(train_upd)
#
# print(session.run(a))
# print(session.run(w))
#
# session.run(train_upd)
#
# print(session.run(a))
# print(session.run(w))
#
# session.run(train_upd)
#
# print(session.run(a))
# print(session.run(w))
#
# session.run(train_upd)
#
# print(session.run(a))
# print(session.run(w))
#

