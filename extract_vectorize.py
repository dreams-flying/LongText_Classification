#! -*- coding: utf-8 -*-
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1' #use GPU with ID=1

import json
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from keras.models import Model
from keras.layers import Lambda
import tensorflow as tf

data_json = './data/iflytek_public/dev_data.json'

class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D):
    """自定义全局池化
    """
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())[:, :, None]
            return K.sum(inputs * mask, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(inputs, axis=1)

#直接使用开源的bert模型
# config_path = './pretrained_model/chinese_L-12_H-768_A-12/albert_config_tiny_g.json'
# checkpoint_path = './pretrained_model/chinese_L-12_H-768_A-12/albert_model.ckpt'
# dict_path = './pretrained_model/chinese_L-12_H-768_A-12/vocab.txt'

#使用经 iflytek_clssification_train.py 训练的模型
config_path = './weights/save_bert/bert_config.json'
checkpoint_path = './weights/save_bert/bert_model.ckpt'
dict_path = './weights/save_bert/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载bert模型，补充平均池化
encoder = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    # model='albert'
)

output = GlobalAveragePooling1D()(encoder.output)
# print(output.shape)
encoder = Model(encoder.inputs, output)

def load_data(filename):
    """加载数据
    返回：[texts]
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            texts = json.loads(l)
            D.append(texts["texts"])
    return D


def predict(texts):
    """句子列表转换为句向量
    """
    batch_token_ids, batch_segment_ids = [], []
    for text in texts:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=128)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
    batch_token_ids = sequence_padding(batch_token_ids)
    batch_segment_ids = sequence_padding(batch_segment_ids)
    outputs = encoder.predict([batch_token_ids, batch_segment_ids])
    return outputs


def convert(data):
    """转换所有样本
    """
    embeddings = []
    for texts in tqdm(data, desc=u'向量化'):
        outputs = predict(texts)
        embeddings.append(outputs)
    embeddings = sequence_padding(embeddings)
    return embeddings


if __name__ == '__main__':

    data_extract_json = data_json[:-5] + '.json'
    print("data_extract_json:", data_extract_json)
    data_extract_npy = data_json[:-5] + '_extract'

    data = load_data(data_extract_json)
    print(len(data))
    embeddings = convert(data)
    np.save(data_extract_npy, embeddings)
    print(u'输出路径：%s.npy' % data_extract_npy)
