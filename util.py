import numpy as np, os
from math import floor, ceil
from TFCommon.DataFeeder import BaseFeeder
from six.moves import xrange
import random

_PAD_ID = 0
_PAD_LOG_ACO = -12

class Feeder(BaseFeeder):
    def __init__(self, *args, **kargs):
        super(Feeder, self).__init__(*args, **kargs)
        self.log_mel_mean = np.expand_dims(np.asarray([self.meta['nancy_log_mel_mean'], self.meta['empha_log_mel_mean']]), axis=1)
        self.log_mel_std = np.expand_dims(np.asarray([self.meta['nancy_log_mel_std'], self.meta['empha_log_mel_std']]), axis=1)
        self.log_stftm_mean = np.expand_dims(np.asarray([self.meta['nancy_log_stftm_mean'], self.meta['empha_log_stftm_mean']]), axis=1)
        self.log_stftm_std = np.expand_dims(np.asarray([self.meta['nancy_log_stftm_std'], self.meta['empha_log_stftm_std']]), axis=1)
    
    def read_by_key(self, key):
        try:
            char_input_key = key
            speaker = None
            if key[:3] != 'neu' and key[:3] != 'foc':
                mel_truth_path = os.path.join(self.meta['nancy_mel_root'], key + '.npy')
                stftm_truth_path = os.path.join(self.meta['nancy_stftm_root'], key + '.npy')
                speaker = 0
            else:
                mel_truth_path = os.path.join(self.meta['empha_mel_root'], key + '.npy')
                stftm_truth_path = os.path.join(self.meta['empha_stftm_root'], key + '.npy')
                speaker = 1
        except Exception as e:
            print('[E] in read_by_key : fetch_pathes failed')
        try:
            raw_char_input = self.meta['char_inputs_dic'].get(char_input_key)
        except Exception as e:
            print('[E] in read_by_key : get char failed')
        try:
            char_input = self.parse_char(raw_char_input)
        except Exception as e:
            print('[E] in read_by_key : parse_char failed')
            print('key: {}'.format(char_input_key))
            print('sen: {}'.format(raw_char_input))
        char_len = len(char_input)
        mel_truth = np.load(mel_truth_path)
        stftm_truth = np.load(stftm_truth_path)
        return char_input, char_len, speaker, mel_truth, stftm_truth

    def normalize_mel(self, batch, speaker_batch):
        mean_batch = np.asarray([self.log_mel_mean[speaker] for speaker in speaker_batch])
        std_batch = np.asarray([self.log_mel_std[speaker] for speaker in speaker_batch])
        return (batch - mean_batch) / std_batch

    def normalize_stftm(self, batch, speaker_batch):
        mean_batch = np.asarray([self.log_stftm_mean[speaker] for speaker in speaker_batch])
        std_batch = np.asarray([self.log_stftm_std[speaker] for speaker in speaker_batch])
        return (batch - mean_batch) / std_batch

    def pre_process_batch(self, batch):
        char_input_batch, char_len_batch, speaker_batch, mel_truth_batch, stftm_truth_batch = zip(*batch)
        char_input_max_len = max(char_len_batch)
        char_input_batch = np.asarray([np.pad(item, (0, char_input_max_len - len(item)), mode='constant', constant_values=_PAD_ID) for item in char_input_batch])
        char_len_batch = np.asarray(char_len_batch)
        speaker_batch = np.asarray(speaker_batch, dtype=np.int32)
        acoustic_max_len = ceil(max([len(item) for item in mel_truth_batch]) / 5) * 5
        log_mel_truth_batch = np.asarray([np.pad(np.log(item), ((0, acoustic_max_len - len(item)), (0, 0)), mode='constant', constant_values=_PAD_LOG_ACO) for item in mel_truth_batch])
        log_stftm_truth_batch = np.asarray([np.pad(np.log(item), ((0, acoustic_max_len - len(item)), (0, 0)), mode='constant', constant_values=_PAD_LOG_ACO) for item in stftm_truth_batch])
        norm_mel_truth_batch = self.normalize_mel(log_mel_truth_batch, speaker_batch)
        norm_stftm_truth_batch = self.normalize_stftm(log_stftm_truth_batch, speaker_batch)
        return char_input_batch, char_len_batch, speaker_batch, norm_mel_truth_batch, norm_stftm_truth_batch

    def split_strategy(self, records_lst):
        sorted_records = sorted(records_lst, key=lambda x: len(x[-2]), reverse=True)
        sorted_batches = [sorted_records[idx*self.batch_size:(idx+1)*self.batch_size] for idx in xrange(self.split_nums)]
        random.shuffle(sorted_batches)
        for idx in xrange(self.split_nums):
            yield sorted_batches[idx]

    def fetch_pathes(self, key):
        char_input_key = key
        mel_truth_path = os.path.join(self.meta['mel_root'], key + '.npy')
        stftm_truth_path = os.path.join(self.meta['stftm_root'], key + '.npy')
        return char_input_key, mel_truth_path, stftm_truth_path

    def parse_char(self, char_seq):
        return [self.meta['char2id_dic'].get(item) for item in char_seq]
