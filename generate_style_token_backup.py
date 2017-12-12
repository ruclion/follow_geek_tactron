import tensorflow as tf, pickle as pkl, os
from Tacotron.Modules.CBHG import CBHG
from TFCommon.Model import Model
from TFCommon.RNNCell import GRUCell
from tensorflow.contrib.rnn import MultiRNNCell, ResidualWrapper
from tensorflow.python.ops import array_ops
from TFCommon.Attention import BahdanauAttentionModule as AttentionModule
from TFCommon.Layers import EmbeddingLayer
import audio, numpy as np
import scipy.io as sio

bidirectional_dynamic_rnn = tf.nn.bidirectional_dynamic_rnn

sr = 16000

EPOCHS = 1000000	# 7142 -> 2M
EMBED_CLASS = 100
EMBED_DIM = 256
SPC_EMBED_CLASS = 5
SPC_EMBED_DIM = 32
ATT_RNN_SIZE = 256
DEC_RNN_SIZE = 256
OUTPUT_MEL_DIM = 80	# 80
OUTPUT_SPEC_DIM = 513 # 513
MAX_OUT_STEPS = 600

class TTS(Model):

    def __init__(self, r=5, is_training=True, name="TTS"):
        super(TTS, self).__init__(name)
        self.__r = r
        self.training = is_training

    @property
    def r(self):
        return self.__r

    def build(self, inp, inp_mask, speaker):
        batch_size = tf.shape(inp)[0]
        input_time_steps = tf.shape(inp)[1]

        ### Encoder [ begin ]
        with tf.variable_scope("encoder"):

            with tf.variable_scope("embedding"):
                embed_inp = EmbeddingLayer(EMBED_CLASS, EMBED_DIM)(inp)

            with tf.variable_scope("pre-net"):
                pre_ed_inp = tf.layers.dropout(tf.layers.dense(embed_inp, 256, tf.nn.relu), training=self.training)
                pre_ed_inp = tf.layers.dropout(tf.layers.dense(pre_ed_inp, 128, tf.nn.relu), training=self.training)

            with tf.variable_scope("CBHG"):
                # batch major
                encoder_output = CBHG(16, (128, 128))(pre_ed_inp, sequence_length=inp_mask, is_training=self.training, time_major=False)

        with tf.variable_scope("attention"):
            att_module = AttentionModule(ATT_RNN_SIZE, encoder_output, sequence_length=inp_mask, time_major=False)

        with tf.variable_scope("decoder"):
            with tf.variable_scope("attentionRnn"):
                att_cell = GRUCell(ATT_RNN_SIZE)
                att_embed_speaker = EmbeddingLayer(SPC_EMBED_CLASS, SPC_EMBED_DIM)(speaker)
            with tf.variable_scope("acoustic_module"):
                aco_cell = MultiRNNCell([ResidualWrapper(GRUCell(DEC_RNN_SIZE)) for _ in range(2)])
                #aco_embed_speaker = EmbeddingLayer(SPC_EMBED_CLASS, SPC_EMBED_DIM)(speaker)

            ### prepare output alpha TensorArray
            reduced_time_steps = tf.div(MAX_OUT_STEPS, self.r)
            att_cell_state = att_cell.init_state(batch_size, tf.float32)
            aco_cell_state = aco_cell.zero_state(batch_size, tf.float32)
            state_tup = tuple([att_cell_state, aco_cell_state])
            output_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
            alpha_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
            init_indic = tf.zeros([batch_size, OUTPUT_MEL_DIM])
            # init_context = tf.zeros((batch_size, 256))

            time = tf.constant(0, dtype=tf.int32)
            cond = lambda time, *_: tf.less(time, reduced_time_steps)
            def body(time, indic, output_ta, alpha_ta, state_tup):
                with tf.variable_scope("att-rnn"):
                    pre_ed_indic = tf.layers.dropout(tf.layers.dense(indic, 256, tf.nn.relu), training=self.training)
                    pre_ed_indic = tf.layers.dropout(tf.layers.dense(pre_ed_indic, 128, tf.nn.relu), training=self.training)
                    att_cell_out, att_cell_state = att_cell(tf.concat([pre_ed_indic, att_embed_speaker], axis=-1), state_tup[0])
                with tf.variable_scope("attention"):
                    query = att_cell_state[0]    # att_cell_out
                    context, alpha = att_module(query)
                    alpha_ta = alpha_ta.write(time, alpha)
                with tf.variable_scope("acoustic_module"):
                    aco_input = tf.layers.dense(tf.concat([att_cell_out, att_embed_speaker, context], axis=-1), DEC_RNN_SIZE)
                    aco_cell_out, aco_cell_state = aco_cell(aco_input, state_tup[1])
                    dense_out = tf.reshape(
                            tf.layers.dense(aco_cell_out, OUTPUT_MEL_DIM * self.r),
                            shape=(batch_size, self.r, OUTPUT_MEL_DIM))
                    output_ta = output_ta.write(time, dense_out)
                    new_indic = dense_out[:, -1]
                state_tup = tuple([att_cell_state, aco_cell_state])

                return tf.add(time, 1), new_indic, output_ta, alpha_ta, state_tup

            ### run loop
            _, _, output_mel_ta, final_alpha_ta, *_ = tf.while_loop(cond, body, [time, init_indic, output_ta, alpha_ta, state_tup])

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

        with tf.variable_scope("loss_and_metric"):
            self.alpha_img = tf.expand_dims(final_alpha, -1)

    def summary(self, suffix, num_img=2):
        sums = []
        sums.append(tf.summary.image("%s/alpha" % suffix, self.alpha_img[:num_img]))

        return tf.summary.merge(sums)



with tf.variable_scope("data"):
    inp = tf.placeholder(name="input", shape=(None, None), dtype=tf.int32)
    inp_mask = tf.placeholder(name="inp_mask", shape=(None,), dtype=tf.int32)
    speaker = tf.placeholder(name='speaker', shape=(None,), dtype=tf.int32)

with tf.variable_scope("model"):
    model = TTS(r=5, is_training=False)
    model.build(inp, inp_mask, speaker)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    saver = tf.train.Saver()


save_path = "save"
model_name = "TTS"

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
        speaker = 0


        speaker = np.asarray([int(speaker)], dtype=np.int32)
        toks = np.asarray([[char2id_dic(c) for c in utt_str]], dtype=np.int32)
        len_arr = np.asarray([len(utt_str)], dtype=np.int32)
        print(speaker, toks, len_arr)
        return speaker, toks, len_arr

    generate_path = 'generate'
    generate_wav_path = os.path.join(generate_path, 'wav')
    if not os.path.exists(generate_path):
        os.makedirs(generate_wav_path)


    with tf.Session() as sess:
        model.sess = sess
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        ckpt = tf.train.get_checkpoint_state(save_path)

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
            speaker_code, tok, lenar = utt2tok(got_utt)
            feed_dict = {inp: tok, inp_mask: lenar, speaker:speaker_code}
            print("Generating ...")
            pred_out = sess.run(model.out_stftm, feed_dict=feed_dict)
            
            # de-emphasis defined in audio.py

            pred_audio = audio.invert_spectrogram(pred_out, 1.5)
            sio.wavfile.write(os.path.join(generate_wav_path, "%d.wav" % cnt), sr, pred_audio)

            sio.wavfile.write(os.path.join(generate_wav_path, "%d.wav" % cnt), sr, pred_audio)
            print("Done!")
            cnt += 1
            '''
            with open(cnt_path, "wb") as f:
                pkl.dump(cnt, f)
            '''
            got_utt = input("Enter to syn:\n")
