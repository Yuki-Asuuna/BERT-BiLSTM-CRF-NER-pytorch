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
            self.u = nn.Parameter(torch.Tensor(1, rnn_dim_audio * 2))
            self.v = nn.Parameter(torch.Tensor(1, rnn_dim_text * 2))
            self.b = nn.Parameter(torch.tensor(1.0))
            # Initialize
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
            f = mfcc39(a, winlen=0.05, winstep=0.05, nfilt=13, nfft=1024, max_frame_length=200)
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
            # a_j_i=tanh(u*s_i+v*h_j+b)
            # s_i => (1,rnn_dim_audio*2),h_j => (1,rnn_dim_text*2)
            batch_num = sequence_output_text.size(0)
            frame_num = sequence_output_audio.size(1)
            word_num = sequence_output_text.size(1)
            alpha = torch.randn((word_num, frame_num))
            for k in range(batch_num):
                for j in range(word_num):
                    for i in range(frame_num):
                        alpha[j][i] = torch.mm(sequence_output_audio[k][i].unsqueeze(0), self.u.t()) + torch.mm(sequence_output_text[k][j].unsqueeze(0),self.v.t()) + self.b

            alpha = torch.tanh(alpha)
            alpha = nn.functional.softmax(alpha)

            # 计算第j个字所对应的语音特征
            new_h = torch.zeros(batch_num, word_num,sequence_output_audio.size(-1))
            for k in range(batch_num):
                for j in range(word_num):
                    for i in range(frame_num):
                        new_h[k][j] += alpha[j][i] * sequence_output_audio[k][i]

            fuse = torch.cat((new_h,sequence_output_text),dim=-1)
            fused_output,_ = self.birnn_fuse(fuse)

            fused_output = self.dropout(fused_output)
            emissions = self.hidden2tag(fused_output)

            return emissions
        else:
            sequence_output_text = self.dropout(sequence_output_text)
            emissions = self.hidden2tag(sequence_output_text)
            return emissions

    def predict(self, input_ids, token_type_ids=None, input_mask=None, audio_data=None):
        audio_feat = self.audio_feat_extract(audio_data)
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask, audio_feat)
        return self.crf.decode(emissions, input_mask.byte())
