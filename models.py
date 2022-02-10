import torch
import torch.nn as nn
import torch.functional as F
from torchcrf import CRF

from pytorch_transformers import BertPreTrainedModel, BertModel
from utils import mfcc39
import numpy as np
from typing import Tuple


# class AdditiveAttention(nn.Module):
#     """
#      Applies a additive attention (bahdanau) mechanism on the output features from the decoder.
#      Additive attention proposed in "Neural Machine Translation by Jointly Learning to Align and Translate" paper.
#      Args:
#          hidden_dim (int): dimesion of hidden state vector
#      Inputs: query, value
#          - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
#          - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.
#      Returns: context, attn
#          - **context**: tensor containing the context vector from attention mechanism.
#          - **attn**: tensor containing the alignment from the encoder outputs.
#      Reference:
#          - **Neural Machine Translation by Jointly Learning to Align and Translate**: https://arxiv.org/abs/1409.0473
#     """
#
#     def __init__(self, hidden_dim: int) -> None:
#         super(AdditiveAttention, self).__init__()
#         self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
#         self.score_proj = nn.Linear(hidden_dim, 1)
#
#     def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
#         attn = F.softmax(score, dim=-1)
#         context = torch.bmm(attn.unsqueeze(1), value)
#         return context, attn

# Additive-Attention(q,k) = W_v * tanh( W_q * q + W_k * k )
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, **kwargs):  # 转换为num_hiddens维度，词向量长度
        # 假设：query：(2, 1, 20), key：(2, 10, 2), value： (2, 10, 4)
        # batch seq word_embedding，  key和value seq_len是一样的，query是一个单独的向量，1×20
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)

    # queries [batch_size,word_num,rnn_dim_text]
    # keys [batch_size,frame_num,rnn_dim_audio]
    # values [batch_size,frame_num,rnn_dim_audio]
    def forward(self, queries, keys, values):
        queries, keys = self.W_q(queries), self.W_k(keys)  # 映射到相同维度 [2,1,8] [2,10,8]
        # query增加一个维度为了方便和key相加。key增加一个维度后面需要
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # torch.Size([2, 1, 1, 8]) torch.Size([2, 1, 10, 8])
        # 此处的相加是指将[1,8]每一行都加到keys的每一行上，每行的长度都为8，且为行向量，才可以相加，所以需要做squeeze去强行创建行向量；如果是两个矩阵，则必须维度相同才可以相加
        # features = torch.Size([2, 1, 10, 8])
        features = torch.tanh(features)

        scores = self.w_v(features)
        # torch.Size([2, 1, 10, 1])
        scores = scores.squeeze(-1)
        # torch.Size([2, 1, 10]) [batch_size, word_num, frame_num]

        self.attention_weigths = nn.functional.softmax(scores)
        # [batch_size, word_num, frame_num]

        # attention weights和values加权相加
        # [batch_size,frame_num,rnn_dim_audio]
        return torch.bmm(self.attention_weigths, values)
        # return [batch_size,word_num,rnn_dim_audio]


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
            # rnn_dim_text 默认256
            out_dim = rnn_dim_text * 2

        # 是否需要加入音频
        if need_audio:
            # 语音LSTM层
            self.birnn_audio = nn.LSTM(39, rnn_dim_audio, num_layers=1, bidirectional=True, batch_first=True)
            # rnn_dim_audio 默认 256

            # Attention Layer Definition
            self.attention_layer = AdditiveAttention(rnn_dim_text * 2, rnn_dim_audio * 2, rnn_dim_audio * 2)
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

        if self.need_audio:
            attn = self.attention_layer(sequence_output_text, sequence_output_audio, sequence_output_audio)
            fuse = torch.cat((sequence_output_text, attn), dim=-1)
            sequence_output_fused, _ = self.birnn_fuse(fuse)
            sequence_output_fused = self.dropout(sequence_output_fused)
            emissions = self.hidden2tag(sequence_output_fused)
            return emissions

        else:
            sequence_output_text = self.dropout(sequence_output_text)
            emissions = self.hidden2tag(sequence_output_text)
            return emissions

    def predict(self, input_ids, token_type_ids=None, input_mask=None, audio_data=None):
        audio_feat = None
        if self.need_audio:
            audio_feat = self.audio_feat_extract(audio_data)
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask, audio_feat)
        return self.crf.decode(emissions, input_mask.byte())
