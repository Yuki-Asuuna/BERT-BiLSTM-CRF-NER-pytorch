import torch
import torch.nn as nn
import torch.functional as F
from TorchCRF import CRF

from pytorch_transformers import BertPreTrainedModel, BertModel
from utils import mfcc39
import numpy as np


class BERT_BiLSTM_CRF(BertPreTrainedModel):

    def __init__(self, config, need_text=False, rnn_dim_text=128, need_audio=False, rnn_dim_audio=128,
                 rnn_dim_fused=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)

        self.num_tags = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size
        self.need_text = need_text
        self.need_audio = need_audio

        # 如果为False，则不要BiLSTM层
        if need_text:
            self.birnn_text = nn.LSTM(config.hidden_size, rnn_dim_text, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim_text * 2

        # 是否需要加入音频
        if need_audio:
            self.birnn_audio = nn.LSTM(39, rnn_dim_audio, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim_fused * 2

        self.hidden2tag = nn.Linear(out_dim, config.num_labels)
        self.crf = CRF(config.num_labels)

    def audio_feat_extract(self, audio_data):
        ret = []
        for a in audio_data:
            f = mfcc39(a, winlen=0.05, winstep=0.03, nfilt=13, nfft=1024, max_frame_length=500)
            ret.append(f)
        return torch.Tensor(np.stack(ret))

    def forward(self, input_ids, tags, token_type_ids=None, input_mask=None, audio_data=None):
        # audio_data 为该batch对应的文件列表
        # 获取批次音频数据的mfcc特征
        audio_feat = self.audio_feat_extract(audio_data)
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask, audio_feat)
        loss = -1 * self.crf(emissions, tags, mask=input_mask.byte())

        return loss

    def tag_outputs(self, input_ids, token_type_ids=None, input_mask=None, audio_feat=None):

        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)

        sequence_output = outputs[0]

        if self.need_text:
            sequence_output, _ = self.birnn_text(sequence_output)

        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)

        return emissions

    def predict(self, input_ids, token_type_ids=None, input_mask=None, sentence_ids=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        return self.crf.decode(emissions, input_mask.byte())
