import numpy as np
'''
a = np.arange(24)
a = np.reshape(a, (2, 3, 4))
b = a[:, -1]
print(a)
print(b)
'''

f = np.load('style_token.npz')
f2 = np.load('test_small_data.npz')

data_inp = f2['inp']
data_inp_mask = f2['inp_mask']

print(data_inp)
print(data_inp_mask)
print(f['all_style'])

'''

f = open('tmp.txt','r')
a = f.read()
d = eval(a)

f.close()

print(d[' '])
'''
'''
a = np.array([1, 2, 3])
b = a[[0, 2]]
del a
print(b)
'''
'''
import tensorflow as tf

import numpy as np
import librosa
import os
import scipy.io.wavfile
import audio
import random

styles_kind = 10
style_dim = 2
BATCH_SIZE = 8

single_style_token = tf.reshape(tf.get_variable('style_token', shape=(1, styles_kind, style_dim), dtype=tf.float32), (styles_kind, style_dim))
style_token_list = [single_style_token for i in range(BATCH_SIZE)]
style_token = tf.stack(style_token_list, axis=0)

session = tf.Session()
session.run(tf.global_variables_initializer())

print(style_token)
'''
'''

sr = 24000

global data_all_size
BATCH_SIZE = 8
EPOCHS = 1000000	# 7142 -> 2M
EMBED_CLASS = 100
EMBED_DIM = 256
STYLE_TOKEN_DIM = 2
SPC_EMBED_CLASS = 5
SPC_EMBED_DIM = 32
ATT_RNN_SIZE = 256
DEC_RNN_SIZE = 256
OUTPUT_MEL_DIM = 128	# 128
OUTPUT_SPEC_DIM = 513 # 513
LR_RATE = 0.001
styles_kind = 10
style_dim = 2
train_r = 5
use_same_style = True


data_path = 'data_audioBook.npz'
save_path = "save"
model_name = "TTS"

def get_next_batch_index():
    a = list(range(0, data_all_size))
    random.shuffle(a)
    return np.array(a[0:min(BATCH_SIZE, data_all_size)])


if __name__ == "__main__":


    data = np.load(data_path)
    data_inp = data['inp']
    data_inp_mask = data['inp_mask']
    data_mel_gtruth = data['mel_gtruth']
    data_spec_gtruth = data['spec_gtruth']
    data_speaker = data['speaker']
    data_style = data['style']
    data_all_style = data['all_style']

    global  data_all_size
    data_all_size = data_inp.shape[0]
    batch_index = get_next_batch_index()

    # batch_index = [1]

    print(batch_index)

    batch_inp = data_inp[batch_index]
    batch_inp_mask = data_inp_mask[batch_index]
    batch_mel_gtruth = data_mel_gtruth[batch_index]
    batch_spec_gtruth = data_spec_gtruth[batch_index]
    batch_speaker = data_speaker[batch_index]
    batch_style = data_style[batch_index]

   
    print(batch_inp.shape)
   

    print('22:', batch_mel_gtruth.shape)

    # print('good time:', good_time)
    print('look timestemp:', batch_mel_gtruth.shape)
    print(batch_mel_gtruth)
'''

'''

data_all_size = 4
BATCH_SIZE = 1

b = np.array([[1], [2], [3], [4]])

def get_next_batch_index():
    global data_all_size
    print(data_all_size)
    a = list(range(0, data_all_size))
    random.shuffle(a)
    print('batch list:', a[0:min(BATCH_SIZE, data_all_size)])
    return np.array(a[0:min(BATCH_SIZE, data_all_size)])

print(b[get_next_batch_index()])


'''


'''

data_input = np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [2, 2], [1, 1]]])

input = tf.placeholder(dtype = np.float32, shape=(None, 3, 2))
# initializer = tf.random_uniform_initializer(-1, 1)
# embedding = tf.get_variable(name="embedding", shape=(5, 10),
#                                         initializer=initializer, dtype=tf.float32)
# embedded = tf.nn.embedding_lookup(embedding, input)

t = tf.layers.dropout(tf.layers.dense(input, 20, tf.nn.relu))

session = tf.Session()
session.run(tf.initialize_all_variables())

print(session.run(t, feed_dict={input:data_input}))
print(np.shape(session.run(t, feed_dict={input:data_input})))

'''
'''

a = np.array([[0.5, 0.5], [0.4, 0.4], [0.3, 0.3]])
repeat_a = np.tile(a, (10, 1, 1))
print(repeat_a)
'''
'''
data_path = 'data.npz'
train_r = 5

data = np.load(data_path)
data_inp = data['inp']
data_inp_mask = data['inp_mask']
data_mel_gtruth = data['mel_gtruth']
data_spec_gtruth = data['spec_gtruth']
data_speaker = data['speaker']
data_style = data['style']

orignal_time = data_mel_gtruth.shape[1]
good_time = orignal_time // train_r * train_r
good_time_mel = np.zeros((data_mel_gtruth.shape[0], good_time, data_mel_gtruth.shape[2]))
good_time_spec = np.zeros((data_spec_gtruth.shape[0], good_time, data_spec_gtruth.shape[2]))
print(data_mel_gtruth.shape, good_time_mel.shape)
print('jj:', data_mel_gtruth[0][0][0])
for i in range(data_mel_gtruth.shape[0]):
    for j in range(good_time):
        for z in range(data_mel_gtruth.shape[2]):
            # print(i, j, z)
            good_time_mel[i][j][z] = data_mel_gtruth[i][j][z]
        for z in range(data_spec_gtruth.shape[2]):
            good_time_spec[i][j][z] = data_spec_gtruth[i][j][z]
'''
# good_time = 2501 // 5 * 5
# print(good_time)
#
# a = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])
# b = a[:][0:1][:]
# print(b)
# batch_mel_gtruth = batch_mel_gtruth[:][0:good_time]
# batch_spec_gtruth = batch_spec_gtruth[:][0:good_time]

'''

a = tf.constant(2501)
b = tf.constant(5)
c = tf.div(a, b)
session = tf.Session()
session.run(tf.initialize_all_variables())
print(session.run(c))

a = np.array([0, 1, 2, 3, 4])
b = np.array([2, 4])
c = a[0:3]
# print(c)

path = 'data.npz'
data = np.load(path)

inp = data['mel_gtruth']
print(inp)

'''

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



# a = tf.constant(3, dtype=tf.float32, shape=(5, 3, 2))
# b = tf.constant(5, shape=(1, 3, 2))
# d = tf.nn.softmax(a)
#
# c = tf.reshape(b, (-1, 1, 2))
#
#
# session = tf.Session()
# print(d, session.run(d))

