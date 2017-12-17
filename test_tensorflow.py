import numpy as np
import os
import time

import tensorflow as tf



def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


x = tf.placeholder(name='input', shape=(None, 3), dtype=tf.float32)
w = tf.get_variable('w', (3, 1), dtype=tf.float32)
b = tf.get_variable('b', (1), dtype=tf.float32)
y = tf.sigmoid(tf.matmul(x, w) + b)
variable_summaries(w)

tf.summary.histogram('weight', w)
tf.summary.histogram('biases', b)







with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    in_x = np.ones((2, 3), dtype=np.float32)
    print(in_x)
    summary_writer = tf.summary.FileWriter('test_logs/', sess.graph)
    for i in range(10):

        out_y = sess.run(y, feed_dict={x:in_x})

        merged_summary_op = tf.summary.merge_all()



        summary_str = sess.run(merged_summary_op)
        summary_writer.add_summary(summary_str, i)


    # tensor_t = tf.convert_to_tensor(t, dtype=tf.float32)
    # print('t is:', tensor_t)
    # print('t val:', sess.run(tensor_t))


    # a = tensor_t

    # print('temp', sess.run(temp))


'''
x = time.time()
print(time.time())

a = 0
for i in range(1000000):
    a += 1
    a /= 2

y = time.time()

print(y)
print('imp:', y - x)
'''

'''

slice_data_size = 256
BATCH_SIZE = 32
slice_num = 23
batch_id = 0
batch_no = 0
batch_idx = None
global_data = None
train_r = 5
OUTPUT_MEL_DIM = 128	# 128
OUTPUT_SPEC_DIM = 513 # 513





def get_next_batch_index():
    global slice_data_size, BATCH_SIZE, slice_num, batch_id, batch_no, batch_idx

    data_path = 'data_audioBook.npz'
    pre_folder = 'big_data'
    if not os.path.exists(pre_folder):
        pre_folder = 'F:\\big_data'

    if batch_no == slice_data_size:
        batch_no = 0
        batch_id = (batch_id + 1) % slice_num

    if batch_no == 0:
        batch_idx = np.random.permutation(slice_data_size)
        print(batch_idx)
        global global_data
        if global_data is not None:
            global_data.close()
        batch_data_path = 'id' + str(batch_id) + data_path
        print(os.path.join(pre_folder, batch_data_path))
        global_data = np.load(os.path.join(pre_folder, batch_data_path))

    zero_data_inp = global_data['inp'][batch_idx[batch_no:batch_no + BATCH_SIZE]]
    data_inp_mask = global_data['inp_mask'][batch_idx[batch_no:batch_no + BATCH_SIZE]]
    data_timestamp = global_data['timestamp'][batch_idx[batch_no:batch_no + BATCH_SIZE]]
    zero_data_mel_gtruth = global_data['mel_gtruth'][batch_idx[batch_no:batch_no + BATCH_SIZE]]
    zero_data_spec_gtruth = global_data['spec_gtruth'][batch_idx[batch_no:batch_no + BATCH_SIZE]]
    data_speaker = global_data['speaker'][batch_idx[batch_no:batch_no + BATCH_SIZE]]

    time_len = np.max(data_timestamp)
    print('original now time:', time_len)
    time_len = (time_len + train_r - 1) // train_r * train_r

    txt_len = np.max(data_inp_mask)
    print(txt_len)

    data_mel_gtruth = zero_data_mel_gtruth[:,0:time_len, :]
    data_spec_gtruth = zero_data_spec_gtruth[:, 0:time_len, :]
    data_inp = zero_data_inp[:, 0:txt_len]

    # a_data_mel_gtruth = np.zeros((BATCH_SIZE, time_len, OUTPUT_MEL_DIM), dtype=np.float32)
    # a_data_spec_gtruth = np.zeros((BATCH_SIZE, time_len, OUTPUT_SPEC_DIM), dtype=np.float32)
    # a_data_inp = np.zeros((BATCH_SIZE, txt_len), dtype=np.int32)
    #
    #
    # for i in range(BATCH_SIZE):
    #     a_data_mel_gtruth[i] = zero_data_mel_gtruth[i][0:time_len]
    #     a_data_spec_gtruth[i] = zero_data_spec_gtruth[i][0:time_len]
    #     a_data_inp[i] = zero_data_inp[i][0:txt_len]
    #
    # print(a_data_inp)
    batch_no += BATCH_SIZE
    return data_inp, data_inp_mask, data_mel_gtruth, data_spec_gtruth


for i in range(5):
    data_inp, data_inp_mask, data_mel_gtruth, data_spec_gtruth = get_next_batch_index()
    print('inp', data_inp)
    print('mel', data_mel_gtruth)
'''
'''
import tensorflow as tf

a = tf.truncated_normal(shape=(1, 10, 2), mean=0.5, stddev=0.1, dtype=tf.float32,seed=32, name = 'style_token')
# t = np.random.normal(0.5, 0.1, (1, 10, 2))

# temp = tf.tile(a,(3, 1, 1))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # tensor_t = tf.convert_to_tensor(t, dtype=tf.float32)
    # print('t is:', tensor_t)
    # print('t val:', sess.run(tensor_t))


    # a = tensor_t
    print(a)
    print('a:', sess.run(a))
    # print('temp', sess.run(temp))

'''



'''

a = np.arange(24)
a = np.reshape(a, (2, 3, 4))

print(a)

print('fff')
b = np.zeros((2, 2, 4))
for i in range(2):
    b[i] = a[i][0:2]
print(b)
'''
'''

f = np.load('style_token.npz')
f2 = np.load('test_small_data.npz')

data_inp = f2['inp']
data_inp_mask = f2['inp_mask']

print(data_inp)
print(data_inp_mask)
print(f['all_style'])

'''
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

