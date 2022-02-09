import torch
import torch.nn as nn
import torch.functional as F
from torchcrf import CRF

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

        # 是否需要文本LSTM；若为否，则退化为baseline
        if need_text:
            # hidden_size默认为768，bert字向量输出维度
            self.birnn_text = nn.LSTM(config.hidden_size, rnn_dim_text, num_layers=1, bidirectional=True,
                                      batch_first=True)
            out_dim = rnn_dim_text * 2

        # 是否需要加入音频
        if need_audio:
            # 语音LSTM层
            self.birnn_audio = nn.LSTM(39, rnn_dim_audio, num_layers=1, bidirectional=True, batch_first=True)
            # Attention Layer Definition
            self.u = nn.Parameter(torch.Tensor(rnn_dim_audio))
            self.v = nn.Parameter(torch.Tensor(rnn_dim_text))
            self.b = nn.Parameter(torch.tensor(1.0))
            nn.init.uniform_(self.u, -0.1, 0.1)
            nn.init.uniform_(self.v, -0.1, 0.1)
            # Attention Layer Definition End

            # 多模态融合层
            self.birnn_fuse = nn.LSTM(rnn_dim_text * 2 + rnn_dim_audio * 2, rnn_dim_fused, num_layers=1,
                                      bidirectional=True,
                                      batch_first=True)
            out_dim = rnn_dim_fused * 2

        self.hidden2tag = nn.Linear(out_dim, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

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

        sequence_output_text = outputs[0]

        # 文本LSTM
        if self.need_text:
            sequence_output_text, _ = self.birnn_text(sequence_output_text)

        # 语音LSTM
        if self.need_audio:
            sequence_output_audio, _ = self.birnn_audio(audio_feat)

        sequence_output_text = self.dropout(sequence_output_text)
        emissions = self.hidden2tag(sequence_output_text)

        return emissions

    def predict(self, input_ids, token_type_ids=None, input_mask=None, audio_data=None):
        audio_feat = self.audio_feat_extract(audio_data)
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask, audio_feat)
        return self.crf.decode(emissions, input_mask.byte())
