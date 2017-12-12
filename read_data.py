import os

import numpy as np
import random
import sys
import audio
import scipy.io as sio

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
sr = 24000

global data_all_size
BATCH_SIZE = 256
EPOCHS = 1000000	# 7142 -> 2M
EMBED_CLASS = 100
EMBED_DIM = 256
STYLE_TOKEN_DIM = 2
SPC_EMBED_CLASS = 5
SPC_EMBED_DIM = 32
ATT_RNN_SIZE = 256
STYLE_ATT_RNN_SIZE = 2
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
data_file_slice_num = 1

def get_next_batch_index(id):
    print('this 000')
    sys.stdout.flush()
    # id = random.randint(0, data_file_slice_num - 1)
    batch_data_path = 'id' + str(id) + data_path
    pre_folder = 'big_data'
    if not os.path.exists(pre_folder):
        pre_folder = 'F:\\big_data'

    print('this 111:', id)
    sys.stdout.flush()

    data = np.load(os.path.join(pre_folder, batch_data_path))
    data_inp = data['inp']
    data_inp_mask = data['inp_mask']
    data_mel_gtruth = data['mel_gtruth']
    data_spec_gtruth = data['spec_gtruth']
    data_speaker = data['speaker']
    data_style = data['style']

    global data_all_size
    data_all_size = data_inp.shape[0]

    a = list(range(0, data_all_size))
    random.shuffle(a)
    batch_index = np.array(a[0:BATCH_SIZE])



    batch_inp = data_inp[batch_index]
    batch_inp_mask = data_inp_mask[batch_index]
    batch_mel_gtruth = data_mel_gtruth[batch_index]
    batch_spec_gtruth = data_spec_gtruth[batch_index]
    batch_speaker = data_speaker[batch_index]
    batch_style = data_style[batch_index]

    return batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth, batch_speaker, batch_style
for i in range(0, 5):
    batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth, batch_speaker, batch_style = get_next_batch_index(i)
    for j in range(BATCH_SIZE):
        if batch_inp_mask[j] == 0:
            print('hhhh???')

'''
np.savez('small_data.npz', inp=batch_inp[0:4], inp_mask=batch_inp_mask[0:4], mel_gtruth=batch_mel_gtruth[0:4], spec_gtruth=batch_spec_gtruth[0:4],
             speaker=batch_speaker[0:4], style=batch_style[0:4], all_style=batch_style)
'''
'''
for i in range(8):
    one_spec = batch_spec_gtruth[i]
    pred_audio = audio.invert_spectrogram(one_spec, 1.5)
    sio.wavfile.write(os.path.join('generate', "%d.wav" % i), sr, pred_audio)
'''