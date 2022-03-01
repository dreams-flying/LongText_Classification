#! -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=1

import json
from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense


maxlen = 128
batch_size = 16

config_path = 'pretrained_model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'pretrained_model/chinese_L-12_H-768_A-12/vocab.txt'

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

def load_data(filename):
    """加载数据
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            l = json.loads(line)
            label = label2id[l["label_des"]]
            D.append((l["sentence"], int(label)))
    return D


# 加载数据集
train_data = load_data('data/iflytek_public/raw_data/train.json')
valid_data = load_data('data/iflytek_public/raw_data/dev.json')


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    # model='albert',
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(
    loss='sparse_categorical_crossentropy',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
        1000: 1,
        2000: 0.1
    }),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """自动保存最新模型
       model.save_weights("best_model.weights") 保存的模型，用 model.load_weights 加载
       bert.save_weights_as_checkpoint("bert_model.ckpt") 保存的模型，用 bert.load_weights_from_checkpoint 加载
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('./weights/best_model.weights')
            bert.save_weights_as_checkpoint('./weights/save_bert/bert_model.ckpt')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )