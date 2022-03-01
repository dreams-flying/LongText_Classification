#! -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=1
import json
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import LayerNormalization
from bert4keras.optimizers import Adam
from keras.layers import *
from keras.models import Model

# 配置信息
input_size = 768
hidden_size = 256
epochs = 100
batch_size = 16
threshold = 0.2

data_json = './data/iflytek_public/train_data.json'
data_extract_json = data_json[:-5] + '.json'
data_extract_npy = data_json[:-5] + '_extract.npy'

data_json2 = './data/iflytek_public/dev_data.json'
data_extract_json2 = data_json2[:-5] + '.json'
data_extract_npy2 = data_json2[:-5] + '_extract.npy'

def load_data(filename):
    """加载数据
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            D.append(json.loads(l))
    return D

label2id = {}
id2label = {}

with open('./data/iflytek_public/raw_data/labels.json', encoding='utf-8') as f:
    for line in f:
        l = json.loads(line)
        if l["label_des"] not in label2id:
            id2label[len(label2id)] = l["label_des"]
            label2id[l["label_des"]] = len(label2id)
num_classes = len(label2id)
# print(num_classes)    #119
# print("label2id:", label2id)

class ResidualGatedConv1D(Layer):
    """门控卷积
    """
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(ResidualGatedConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.supports_masking = True

    def build(self, input_shape):
        super(ResidualGatedConv1D, self).build(input_shape)
        self.conv1d = Conv1D(
            filters=self.filters * 2,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='same',
        )
        self.layernorm = LayerNormalization()

        if self.filters != input_shape[-1]:
            self.dense = Dense(self.filters, use_bias=False)

        self.alpha = self.add_weight(
            name='alpha', shape=[1], initializer='zeros'
        )

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs = inputs * mask[:, :, None]

        outputs = self.conv1d(inputs)
        gate = K.sigmoid(outputs[..., self.filters:])
        outputs = outputs[..., :self.filters] * gate
        outputs = self.layernorm(outputs)

        if hasattr(self, 'dense'):
            inputs = self.dense(inputs)

        return inputs + self.alpha * outputs

    def compute_output_shape(self, input_shape):
        shape = self.conv1d.compute_output_shape(input_shape)
        return (shape[0], shape[1], shape[2] // 2)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate
        }
        base_config = super(ResidualGatedConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


x_in = Input(shape=(None, input_size))
x = x_in

x = Masking()(x)
x = Dropout(0.1)(x)
x = Dense(hidden_size, use_bias=False)(x)
x = Dropout(0.1)(x)
x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
x = Dropout(0.1)(x)
x = LSTM(80)(x)
x = Dropout(0.1)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(x_in, x)
model.compile(
    loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy']
)
# model.summary()


def evaluate(data, data_x):
    """验证集评估
    """
    total, right = 0., 0.
    y_pred = model.predict(data_x).argmax(axis=1)
    total = len(y_pred)
    for d, yp in tqdm(zip(data, y_pred)):
        y_true = int(label2id[d["label_des"]])
        right += (y_true == yp).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """训练回调
    """
    def __init__(self):
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_data, valid_x)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('./weights/best_model_clssification.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


def data_split(data, mode):
    """划分训练集和验证集
    """
    D = [d for i, d in enumerate(data)]
    if isinstance(data, np.ndarray):
        return np.array(D)
    else:
        return D

if __name__ == '__main__':

    # 加载数据
    data1 = load_data(data_extract_json)
    data2 = load_data(data_extract_json2)
    data_x1 = np.load(data_extract_npy)
    data_x2 = np.load(data_extract_npy2)
    tempList = []
    for i, d in enumerate(data1):
        tempList.append([int(label2id[d["label_des"]])])
    data_y1 = np.array(tempList)

    valid_data = data_split(data2, 'valid')
    train_x = data_split(data_x1, 'train')
    valid_x = data_split(data_x2, 'valid')
    train_y = data_split(data_y1, 'train')

    # 启动训练
    evaluator = Evaluator()

    model.fit(
        train_x,
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[evaluator]
    )
