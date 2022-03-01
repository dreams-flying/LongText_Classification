#! -*- coding: utf-8 -*-
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=1
import json

from bert4keras.snippets import sequence_padding
from extract_model import model, load_data, data_split
from extract_vectorize import predict
import numpy as np

label2id = {}
id2label = {}
with open('./data/iflytek_public/raw_data/labels.json', encoding='utf-8') as f:
    for line in f:
        l = json.loads(line)
        if l["label_des"] not in label2id:
            id2label[len(label2id)] = l["label_des"]
            label2id[l["label_des"]] = len(label2id)
num_classes = len(label2id)

def get_split(text, maxlen=120, overlap=40):
    partlen = maxlen - overlap
    l_total = []
    # print(len(text))
    if len(text) // partlen > 0:
        n = len(text) // partlen
    else:
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = text[:maxlen]
            l_total.append("".join(l_parcial))
        else:
            l_parcial = text[w * partlen:w * partlen + maxlen]
            l_total.append("".join(l_parcial))
    return l_total

def evaluate(data, data_x):
    """验证集评估
    """
    total, right = 0., 0.
    y_pred = model.predict(data_x).argmax(axis=1)
    total = len(y_pred)
    for d, yp in zip(data, y_pred):
        y_true = int(label2id[d["label_des"]])
        right += (y_true == yp).sum()
    return right / total

def convert(data):
    """转换样本
    """
    embeddings = []
    outputs = predict(data)
    embeddings.append(outputs)
    embeddings = sequence_padding(embeddings)
    return embeddings


def predict_to_file():
    """输出测试结果到文件
        结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    fw = open("iflytek_predict.json", "w", encoding="utf-8")
    with open("data/iflytek_public/raw_data/test.json", encoding="utf-8") as f:
        for line in f:
            tempDict = {}
            l = json.loads(line)
            texts = get_split(l["sentence"], maxlen=120, overlap=40)
            embeddings = convert(texts)
            y_pred = model.predict(embeddings).argmax(axis=1)
            tempDict["id"] = l["id"]
            tempDict["label"] = str(y_pred[0])
            l1 = json.dumps(tempDict, ensure_ascii=False)
            fw.write(l1 + '\n')
    fw.close()

if __name__ == '__main__':

    model.load_weights("weights/best_model_clssification.weights")

    #预测到文件
    predict_to_file()

    #评估
    # data_json = './data/iflytek_public/dev_data.json'
    # data_extract_json = data_json[:-5] + '.json'
    # data_extract_npy = data_json[:-5] + '_extract.npy'
    # data = load_data(data_extract_json)
    # data_x = np.load(data_extract_npy)
    # valid_data = data_split(data, 'valid')
    # valid_x = data_split(data_x, 'valid')
    # val_acc = evaluate(valid_data, valid_x)
    # print(val_acc)

    #测试
    text = "朴朴快送超市创立于2016年，专注于打造移动端30分钟即时配送一站式购物平台，商品品类包含水果、蔬菜、肉禽蛋奶、海鲜水产、粮油调味、酒水饮料、休闲食品、日用品、外卖等。朴朴公司希望能以全新的商业模式，更高效快捷的仓储配送模式，致力于成为更快、更好、更多、更省的在线零售平台，带给消费者更好的消费体验，同时推动中国食品安全进程，成为一家让社会尊敬的互联网公司。,朴朴一下，又好又快,1.配送时间提示更加清晰友好2.保障用户隐私的一些优化3.其他提高使用体验的调整4.修复了一些已知bug"

    texts = get_split(text, maxlen=120, overlap=40)
    embeddings = convert(texts)
    print(embeddings.shape)
    y_pred = model.predict(embeddings).argmax(axis=1)
    print(id2label[y_pred[0]])


