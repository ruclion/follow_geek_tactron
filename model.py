import tensorflow as tf, pickle as pkl, os
from util import Feeder
from Tacotron.Modules.CBHG import CBHG
from TFCommon.Model import Model
from TFCommon.RNNCell import GRUCell
from tensorflow.contrib.rnn import MultiRNNCell, ResidualWrapper
from tensorflow.python.ops import array_ops
from TFCommon.Attention import BahdanauAttentionModule as AttentionModule
from TFCommon.Layers import EmbeddingLayer
import audio, numpy as np
import tqdm

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
LR_RATE = 0.001
#LR_RATE = 0.0005
#LR_RATE = 0.00025
#LR_RATE = 0.0001


class TTS(Model):

    def __init__(self, r=5, is_training=True, name="TTS"):
        super(TTS, self).__init__(name)
        self.__r = r
        self.training = is_training

    @property
    def r(self):
        return self.__r

    def build(self, inp, inp_mask, speaker, mel_gtruth, spec_gtruth):
        batch_size = tf.shape(inp)[0]
        input_time_steps = tf.shape(inp)[1]
        output_time_steps = tf.shape(mel_gtruth)[1]

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
                aco_embed_speaker = EmbeddingLayer(SPC_EMBED_CLASS, SPC_EMBED_DIM)(speaker)

            ### prepare output alpha TensorArray
            reduced_time_steps = tf.div(output_time_steps, self.r)
            att_cell_state = att_cell.init_state(batch_size, tf.float32)
            aco_cell_state = aco_cell.zero_state(batch_size, tf.float32)
            state_tup = tuple([att_cell_state, aco_cell_state])
            output_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
            alpha_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
            indic_ta = tf.TensorArray(size=self.r + output_time_steps, dtype=tf.float32)
            time_major_mel_gtruth = tf.transpose(mel_gtruth, perm=(1, 0, 2))
            indic_array = tf.concat([tf.zeros([self.r, batch_size, OUTPUT_MEL_DIM]), time_major_mel_gtruth], axis=0)
            indic_ta = indic_ta.unstack(indic_array)
            #init_context = tf.zeros((batch_size, 256))

            time = tf.constant(0, dtype=tf.int32)
            cond = lambda time, *_: tf.less(time, reduced_time_steps)
            def body(time, output_ta, alpha_ta, state_tup):
                with tf.variable_scope("att-rnn"):
                    pre_ed_indic = tf.layers.dropout(tf.layers.dense(indic_ta.read(self.r*time + self.r - 1), 256, tf.nn.relu), training=self.training)
                    pre_ed_indic = tf.layers.dropout(tf.layers.dense(pre_ed_indic, 128, tf.nn.relu), training=self.training)
                    att_cell_out, att_cell_state = att_cell(tf.concat([pre_ed_indic, att_embed_speaker], axis=-1), state_tup[0])
                with tf.variable_scope("attention"):
                    query = att_cell_state[0]    # att_cell_out
                    context, alpha = att_module(query)
                    alpha_ta = alpha_ta.write(time, alpha)
                with tf.variable_scope("acoustic_module"):
                    aco_input = tf.layers.dense(tf.concat([att_cell_out, att_embed_speaker, context], axis=-1), DEC_RNN_SIZE)
                    aco_cell_out, aco_cell_state = aco_cell(aco_input, state_tup[1])
                    dense_out = tf.layers.dense(aco_cell_out, OUTPUT_MEL_DIM * self.r)
                    output_ta = output_ta.write(time, dense_out)
                state_tup = tuple([att_cell_state, aco_cell_state])

                return tf.add(time, 1), output_ta, alpha_ta, state_tup

            ### run loop
            _, output_mel_ta, final_alpha_ta, *_ = tf.while_loop(cond, body, [time, output_ta, alpha_ta, state_tup])

        ### time major
        with tf.variable_scope("output"):
            output_mel = tf.reshape(output_mel_ta.stack(), shape=(reduced_time_steps, batch_size, OUTPUT_MEL_DIM * self.r))
            output_mel = tf.reshape(tf.transpose(output_mel, perm=(1, 0, 2)), shape=(batch_size, output_time_steps, OUTPUT_MEL_DIM))
            self.out_mel = output_mel

            with tf.variable_scope("post-net"):
                output_post = CBHG(8, (256, OUTPUT_MEL_DIM))(output_mel, sequence_length=None, is_training=self.training, time_major=False)
                output_spec = tf.layers.dense(output_post, OUTPUT_SPEC_DIM)
                self.out_stftm = output_spec

            final_alpha = tf.reshape(final_alpha_ta.stack(), shape=(reduced_time_steps, batch_size, input_time_steps))
            final_alpha = tf.transpose(final_alpha, perm=(1,0,2))    # batch major

        with tf.variable_scope("loss_and_metric"):
            self.loss_mel = tf.reduce_mean(tf.abs(mel_gtruth - output_mel))
            self.loss_spec = tf.reduce_mean(tf.abs(spec_gtruth - output_spec))
            self.loss = self.loss_mel + self.loss_spec
            self.alpha_img = tf.expand_dims(final_alpha, -1)

    def summary(self, suffix, num_img=2):
        sums = []
        sums.append(tf.summary.scalar("%s/loss" % suffix, self.loss))
        sums.append(tf.summary.scalar("%s/loss_mel" % suffix, self.loss_mel))
        sums.append(tf.summary.scalar("%s/loss_spec" % suffix, self.loss_spec))
        sums.append(tf.summary.image("%s/alpha" % suffix, self.alpha_img[:num_img]))

        return tf.summary.merge(sums)

coord = tf.train.Coordinator()

with tf.variable_scope("data"):
    inp = tf.placeholder(name="input", shape=(None, None), dtype=tf.int32)
    inp_mask = tf.placeholder(name="inp_mask", shape=(None,), dtype=tf.int32)
    speaker = tf.placeholder(name='speaker', shape=(None,), dtype=tf.int32)
    mel_gtruth = tf.placeholder(name="output_mel", shape=(None, None, OUTPUT_MEL_DIM), dtype=tf.float32)
    spec_gtruth = tf.placeholder(name="output_spec", shape=(None, None, OUTPUT_SPEC_DIM), dtype=tf.float32)

with open('../../data/repre_train_meta.pkl', 'rb') as f:
    train_meta = pkl.load(f)
train_feeder = Feeder(coord, [inp, inp_mask, speaker, mel_gtruth, spec_gtruth], train_meta, batch_size=32, split_nums=32)

with open('../../data/repre_dev_meta.pkl', 'rb') as f:
    dev_meta = pkl.load(f)
dev_feeder = Feeder(coord, [inp, inp_mask, speaker, mel_gtruth, spec_gtruth], dev_meta, batch_size=64, split_nums=16)

with tf.variable_scope("model"):
    train_model = TTS(r=5)
    train_model.build(*train_feeder.fed_holders)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_var = tf.trainable_variables()
    with tf.variable_scope("optimizer"):
        opt = tf.train.AdamOptimizer(LR_RATE)
        grads_and_vars = opt.compute_gradients(train_model.loss)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_upd = opt.apply_gradients(grads_and_vars, global_step=global_step)
    saver = tf.train.Saver()

with tf.variable_scope("model", reuse=True):
    dev_model = TTS(r=5, is_training=False)
    dev_model.build(*dev_feeder.fed_holders)

train_summary = train_model.summary("train", 2)

save_path = "save"
model_name = "TTS"

if __name__ == "__main__":
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with tf.Session() as sess:
        train_model.sess = sess
        dev_model.sess = sess
        train_feeder.start_in_session(sess)
        dev_feeder.start_in_session(sess)

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        ckpt = tf.train.get_checkpoint_state(save_path)

        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(save_path, ckpt_name))
        writer = tf.summary.FileWriter("log/train", sess.graph)

        guess_train_batches = 288 #270
        guess_dev_batches = 18 #17
        try:
            epoch_pbar = tqdm.trange(EPOCHS)
            epoch_pbar.set_description('Epoch')
            mean_loss_holder = tf.placeholder(shape=(), dtype=tf.float32, name='mean_loss')
            train_epoch_summary = tf.summary.scalar('epoch/train/loss', mean_loss_holder)
            dev_epoch_summary = tf.summary.scalar('epoch/dev/loss', mean_loss_holder)
            for cnt in epoch_pbar:
                # train phase
                train_bar = tqdm.trange(guess_train_batches)
                train_bar.set_description('Train')
                total_loss = 0.
                for batch_cnt in train_bar:
                    if not coord.should_stop():
                        _, loss_eval, global_step_eval = sess.run([train_upd, train_model.loss, global_step])
                        total_loss += loss_eval
                
                        if global_step_eval % 50 == 0:
                            train_sum_eval = sess.run(train_summary)
                            writer.add_summary(train_sum_eval, global_step_eval)
                        if global_step_eval % 2000 == 0:
                            train_model.save(save_path, global_step_eval)
                        if global_step_eval == 100000:
                            break
                mean_loss = total_loss / guess_train_batches
                with open('train_loss.txt', 'a') as f:
                    f.write('{:f}\n'.format(mean_loss))
                train_epoch_summary_eval = sess.run(train_epoch_summary, feed_dict={mean_loss_holder: mean_loss})
                writer.add_summary(train_epoch_summary_eval, cnt)
            
                # dev phase
                dev_bar = tqdm.trange(guess_dev_batches)
                dev_bar.set_description('Dev')
                total_loss = 0.
                for batch_cnt in dev_bar:
                    if not coord.should_stop():
                        loss_eval = sess.run(dev_model.loss)
                        total_loss += loss_eval
                mean_loss = total_loss / guess_dev_batches
                with open('dev_loss.txt', 'a') as f:
                    f.write('{:f}\n'.format(mean_loss))
                dev_epoch_summary_eval = sess.run(dev_epoch_summary, feed_dict={mean_loss_holder: mean_loss})
                writer.add_summary(dev_epoch_summary_eval, cnt)
                
                if global_step_eval == 100000:
                    break

        except Exception as e:
            print('Training stopped')
            coord.request_stop()
