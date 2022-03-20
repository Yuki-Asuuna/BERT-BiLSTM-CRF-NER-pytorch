import json
import pathlib
import scipy.io.wavfile as wav
import python_speech_features
from python_speech_features import mfcc
import numpy as np
import os
import matplotlib.pyplot as plt

max_frame_len = 0


def readFiles(filePath: str):
    objs = []
    with open(filePath, encoding='utf-8') as f:
        for line in f.read().splitlines():
            # print(line)
            obj = json.loads(line)
            objs.append(obj)

    return objs


def mfcc39(filename):
    fs, wavedata = wav.read(filename)
    mfcc_feature = mfcc(wavedata, fs, winlen=0.05, winstep=0.03, nfilt=13,
                        nfft=1024)  # mfcc系数     # nfilt为返回的mfcc数据维数，默认为13维
    d_mfcc_feat = python_speech_features.base.delta(mfcc_feature,
                                                    1)  # feat 为mfcc数据或fbank数据    # N - N为1代表一阶差分，N为2代表二阶差分     # 返回：一个大小为特征数量的numpy数组，包含有delta特征，每一行都有一个delta向量
    d_mfcc_feat2 = python_speech_features.base.delta(mfcc_feature, 2)
    mfccs = np.hstack((mfcc_feature, d_mfcc_feat, d_mfcc_feat2))
    # 返回 帧数*39 的mfccs参数
    return mfccs

def extraction():
    objs = readFiles('./ori/test.json')

    all_feats = []

    folder_name = ['train', 'dev', 'test']
    for index, obj in enumerate(objs[:]):
        if index % 100 == 0:
            print('(%d/%d)' % (index, len(objs)))
        audio_name = obj['audio']

        prefix = audio_name[6:11]

        full_path = ''
        for fn in folder_name:
            full_path = '/Users/bytedance/data_aishell/wav/' + prefix + '/' + fn + '/' + prefix + '/' + audio_name + '.wav'
            if os.path.exists(full_path) == True:
                break

        feat = mfcc39(full_path)
        if len(feat) > max_frame_len:
            max_frame_len = len(feat)
        all_feats.append(feat)

    print('max_frame_len=', max_frame_len)

    tmp = []
    for feat in all_feats:
        pad_delta = max_frame_len - len(feat)
        tmp.append(np.row_stack((feat, np.zeros((pad_delta, 39)))))
        # print(feat.shape)
        # 补0
        # tmp.append(np.column_stack((feat, np.zeros((39, pad_delta)))))

    res = np.stack(tmp)

    print('shape=', res.shape)
    np.save('audio_feature.npy', res)

def statSentence():
    objs_1 = readFiles('./ori/train.json')
    objs_2 = readFiles('./ori/test.json')
    objs_3 = readFiles('./ori/valid.json')

    x = []
    for obj in objs_1+objs_2+objs_3:
        sentence = obj['sentence']
        x.append(len(sentence))

    nx = np.bincount(x)
    #for (i,cnt) in enumerate(nx):
    #    print(i,cnt)
    plt.bar(range(len(nx)),nx,width=0.3)
    plt.xticks(range(len(nx)))
    #plt.xlabel('句子长度')
    #plt.ylabel('句子条数')
    #plt.title('文本句子长度分布')
    plt.show()

if __name__ == '__main__':
    # statSentence()
    pass



