import tensorflow as tf, pickle as pkl, os
from Tacotron.Modules.CBHG import CBHG
from TFCommon.Model import Model
from TFCommon.RNNCell import GRUCell
from tensorflow.contrib.rnn import MultiRNNCell, ResidualWrapper
from tensorflow.python.ops import array_ops
from TFCommon.Attention import BahdanauAttentionModule as AttentionModule
from TFCommon.Layers import EmbeddingLayer
import numpy as np
import random
import sys
import gc
import scipy.io as sio
import audio
import librosa.display
import librosa
import matplotlib.pyplot as plt

bidirectional_dynamic_rnn = tf.nn.bidirectional_dynamic_rnn

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
sr = 24000

data_all_size = 256


BATCH_SIZE = 1
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
save_path = "save_simple_style"
model_name = "TTS"
start_data_file_slice_num = 0
data_file_slice_num = 5

use_global_data = False
global_data = None
now_id = 3

MAX_OUT_STEPS = 600





generate_path = 'generate'






class TTS(Model):

    def __init__(self, r=5, is_training=True, name="TTS"):
        super(TTS, self).__init__(name)
        self.__r = r
        self.training = is_training

    @property
    def r(self):
        return self.__r

    def build(self, inp, inp_mask):

        batch_size = tf.shape(inp)[0]
        input_time_steps = tf.shape(inp)[1]

        ### Encoder [ begin ]
        with tf.variable_scope("encoder"):
            with tf.variable_scope("embedding"):
                embed_inp = EmbeddingLayer(EMBED_CLASS, EMBED_DIM)(inp)

            with tf.variable_scope("changeToVarible"):
                self.single_style_token = tf.get_variable('style_token', (1, 10, 2), dtype=tf.float32)
                self.style_token = tf.tile(self.single_style_token, (batch_size, 1, 1))

            with tf.variable_scope("pre-net"):
                pre_ed_inp = tf.layers.dropout(tf.layers.dense(embed_inp, 256, tf.nn.relu), training=self.training)
                pre_ed_inp = tf.layers.dropout(tf.layers.dense(pre_ed_inp, 128, tf.nn.relu), training=self.training)

            with tf.variable_scope("CBHG"):
                # batch major
                encoder_output = CBHG(16, (128, 128))(pre_ed_inp, sequence_length=inp_mask, is_training=self.training,
                                                      time_major=False)

        with tf.variable_scope("attention"):
            att_module = AttentionModule(ATT_RNN_SIZE, encoder_output, sequence_length=inp_mask, time_major=False)
        with tf.variable_scope("attention_style"):
            att_module_style = AttentionModule(STYLE_ATT_RNN_SIZE, self.style_token, time_major=False)

        with tf.variable_scope("decoder"):
            with tf.variable_scope("attentionRnn"):
                att_cell = GRUCell(ATT_RNN_SIZE)
            with tf.variable_scope("acoustic_module"):
                aco_cell = MultiRNNCell([ResidualWrapper(GRUCell(DEC_RNN_SIZE)) for _ in range(2)])

            ### prepare output alpha TensorArray
            reduced_time_steps = tf.div(MAX_OUT_STEPS, self.r)
            att_cell_state = att_cell.init_state(batch_size, tf.float32)
            aco_cell_state = aco_cell.zero_state(batch_size, tf.float32)
            state_tup = tuple([att_cell_state, aco_cell_state])
            output_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
            alpha_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
            alpha_style_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
            init_indic = tf.zeros([batch_size, OUTPUT_MEL_DIM])
            # init_context = tf.zeros((batch_size, 256))

            time = tf.constant(0, dtype=tf.int32)
            cond = lambda time, *_: tf.less(time, reduced_time_steps)
            def body(time, indic, output_ta, alpha_ta, alpha_style_ta, state_tup):
                with tf.variable_scope("att-rnn"):
                    pre_ed_indic = tf.layers.dropout(tf.layers.dense(indic, 256, tf.nn.relu), training=self.training)
                    pre_ed_indic = tf.layers.dropout(tf.layers.dense(pre_ed_indic, 128, tf.nn.relu), training=self.training)
                    att_cell_out, att_cell_state = att_cell(tf.concat([pre_ed_indic], axis=-1), state_tup[0])
                with tf.variable_scope("attention"):
                    query = att_cell_state[0]    # att_cell_out
                    context, alpha = att_module(query)
                    alpha_ta = alpha_ta.write(time, alpha)
                with tf.variable_scope("attention_style"):
                    context_style, alpha_style = att_module_style(query)
                    alpha_style_ta = alpha_style_ta.write(time, alpha_style)
                with tf.variable_scope("acoustic_module"):
                    aco_input = tf.layers.dense(tf.concat([att_cell_out, context, context_style], axis=-1), DEC_RNN_SIZE)
                    aco_cell_out, aco_cell_state = aco_cell(aco_input, state_tup[1])
                    dense_out = tf.reshape(
                            tf.layers.dense(aco_cell_out, OUTPUT_MEL_DIM * self.r),
                            shape=(batch_size, self.r, OUTPUT_MEL_DIM))
                    output_ta = output_ta.write(time, dense_out)
                    new_indic = dense_out[:, -1]
                state_tup = tuple([att_cell_state, aco_cell_state])

                return tf.add(time, 1), new_indic, output_ta, alpha_ta, alpha_style_ta, state_tup

            ### run loop
            _, _, output_mel_ta, final_alpha_ta, final_alpha_style_ta, *_ = tf.while_loop(cond, body, [time, init_indic, output_ta, alpha_ta, alpha_style_ta, state_tup])

        ### time major
        with tf.variable_scope("output"):
            output_mel = tf.reshape(output_mel_ta.stack(), shape=(reduced_time_steps, batch_size, OUTPUT_MEL_DIM * self.r))
            output_mel = tf.reshape(tf.transpose(output_mel, perm=(1, 0, 2)), shape=(batch_size, MAX_OUT_STEPS, OUTPUT_MEL_DIM))
            self.out_mel = output_mel

            with tf.variable_scope("post-net"):
                output_post = CBHG(8, (256, OUTPUT_MEL_DIM))(output_mel, sequence_length=None, is_training=self.training, time_major=False)
                output_spec = tf.layers.dense(output_post, OUTPUT_SPEC_DIM)
                self.out_stftm = output_spec

            final_alpha = tf.reshape(final_alpha_ta.stack(), shape=(reduced_time_steps, batch_size, input_time_steps))
            final_alpha = tf.transpose(final_alpha, perm=(1,0,2))    # batch major
            self.alpha_hjk_img = tf.reshape(final_alpha, shape=(batch_size, reduced_time_steps, input_time_steps))

            final_alpha_style = tf.reshape(final_alpha_style_ta.stack(), shape=(reduced_time_steps, batch_size, styles_kind))
            final_alpha_style = tf.transpose(final_alpha_style, perm=(1, 0, 2))  # batch major
            self.alpha_style_hjk_img = tf.reshape(final_alpha_style, shape=(batch_size, reduced_time_steps, styles_kind))

        with tf.variable_scope("loss_and_metric"):
            self.alpha_img = tf.expand_dims(final_alpha, -1)

    def summary(self, suffix, num_img=2):
        sums = []
        sums.append(tf.summary.image("%s/alpha" % suffix, self.alpha_img[:num_img]))

        return tf.summary.merge(sums)


with tf.variable_scope("data"):
    inp = tf.placeholder(name="input", shape=(None, None), dtype=tf.int32)
    inp_mask = tf.placeholder(name="inp_mask", shape=(None,), dtype=tf.int32)

with tf.variable_scope("model"):
    model = TTS(r=train_r)
    model.build(inp, inp_mask)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_var = tf.trainable_variables()
    model.saver = tf.train.Saver()


    '''
    model = TTS(r=5, is_training=False)
    model.build(inp, inp_mask, speaker, tmp_style_token)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    model.saver = tf.train.Saver()
    # train_model.saver = tf.train.Saver()
    '''



if __name__ == "__main__":
    assert os.path.exists(save_path), "No model found!"


    '''
    with open("../../data/repre_train_meta.pkl", "rb") as f:
        train_meta = pkl.load(f)
    char2id_dic = train_meta['char2id_dic']
    log_stftm_mean = [train_meta['nancy_log_stftm_mean'], train_meta['empha_log_stftm_mean']]
    log_stftm_std = [train_meta['nancy_log_stftm_std'], train_meta['empha_log_stftm_std']]
    '''
    '''
    def utt2tok(utt_str):
        speaker, utt_str = utt_str.split('|')
        speaker = np.asarray([int(speaker)], dtype=np.int32)
        toks = np.asarray([[char2id_dic.get(c) for c in utt_str]], dtype=np.int32)
        len_arr = np.asarray([len(utt_str)], dtype=np.int32)
        return speaker, toks, len_arr
    '''
    def char2id_dic(ch):
        with open('tmp.txt', 'r') as f:
            a = f.read()
            d = eval(a)
            return d[ch]


    def utt2tok(utt_str):

        toks = np.array([[char2id_dic(c) for c in utt_str]], dtype=np.int32)
        len_arr = np.array([len(utt_str)], dtype=np.int32)
        print(toks, len_arr)
        for i in range(toks.shape[1]):
            print('--:', toks[0][i])

        '''
                    test!!!! 
        

        speaker = np.asarray([int(0)], dtype=np.int32)

        f2 = np.load('test_small_data.npz')

        data_inp = f2['inp']
        data_inp_mask = f2['inp_mask']
        toks = np.zeros((1, 64), dtype=np.int32)
        for i in range(64):
            toks[0][i] = data_inp[2][i]
        len_arr = np.asarray([int(data_inp_mask[2])], dtype=np.int32)

        
                    test!!!! over!! 
        '''

        return toks, len_arr


    generate_wav_path = os.path.join(generate_path, 'wav')
    if not os.path.exists(generate_path):
        os.makedirs(generate_wav_path)


    with tf.Session() as sess:
        model.sess = sess
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])


        ckpt = tf.train.get_checkpoint_state(save_path)
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            model.saver.restore(sess, os.path.join(save_path, ckpt_name))
            print('restore path:', ckpt_name)

        '''
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(save_path, ckpt_name))
        '''
        global_step_eval = tf.train.global_step(sess, global_step)

        '''
        cnt_path = os.path.join(generate_path, "cnt.pkl")
        if os.path.exists(cnt_path):
            with open(cnt_path, "rb") as f:
                cnt = pkl.load(f)
        else:
            cnt = 0
        '''

        cnt = 0
        got_utt = input("Enter to syn:\n")
        txt_path = os.path.join(generate_path, "texts.txt")
        while got_utt != "EXIT":
            with open(txt_path, "a") as f:
                f.write("%d. %s\n" % (cnt, got_utt))
            tok, lenar = utt2tok(got_utt)

            print("Generating ...")
            # copy_style_token = np.copy(tmp_style_token)
            # copy_style_token[0][0] = 0
            # copy_style_token[0][1] = 0
            # copy_style_token[1][0] = 0
            # copy_style_token[1][1] = 0
            # copy_style_token[2][0] = 0
            # copy_style_token[2][1] = 0
            # copy_style_token[3][0] = 0
            # copy_style_token[3][1] = 0
            pred_out, alpha_hjk_img, alpha_style_hjk_img, style_token = sess.run([model.out_stftm, model.alpha_hjk_img, model.alpha_style_hjk_img, model.single_style_token], feed_dict={inp:tok, inp_mask:lenar})
            
            # de-emphasis defined in audio.py
            print('pre:', style_token)



            pred_audio, exp_spec = audio.invert_spectrogram(pred_out, 1.2)


            # plt.subplot(2, 2, 1)
            # D = librosa.amplitude_to_db(exp_spec, ref=np.max)
            D = np.log(exp_spec)
            plt.matshow(D, cmap='hot')
            plt.show()
            # librosa.display.specshow(D, y_axis='linear')
            # plt.matshow(np.reshape(pred_out, (pred_out.shape[-2], pred_out.shape[-1])), cmap='hot')
            # plt.colorbar(format='%+2.0f dB')
            # plt.title('Linear-frequency power spectrogram')

            # plt.subplot(2, 2, 2)

            alpha_style_hjk_img = np.reshape(alpha_style_hjk_img, (alpha_style_hjk_img.shape[1], alpha_style_hjk_img.shape[2]))
            plt.matshow(alpha_style_hjk_img.T, cmap='hot')

            plt.show()

            alpha_hjk_img = np.reshape(alpha_hjk_img, (alpha_hjk_img.shape[1], alpha_hjk_img.shape[2]))
            plt.matshow(alpha_hjk_img.T, cmap='hot')

            plt.show()

            sio.wavfile.write(os.path.join(generate_wav_path, "%d.wav" % cnt), sr, pred_audio)


            print("Done!")
            cnt += 1
            '''
            with open(cnt_path, "wb") as f:
                pkl.dump(cnt, f)
            '''
            got_utt = input("Enter to syn:\n")
