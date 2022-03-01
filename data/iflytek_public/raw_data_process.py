#! -*- coding: utf-8 -*-
import json
# from data import tokenization
import random


def load_data(filename):
    """加载数据
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            l = json.loads(line)
            D.append(l)
    return D

def get_split(text, maxlen=120, overlap=40):
    partlen = maxlen - overlap
    l_total = []
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


data = load_data('raw_data/train.json')
random.shuffle(data)  # 随机一下

fw = open("train_data.json", 'w', encoding='utf-8')

for i in range(len(data)):
    totalDict = {}
    totalDict["label"] = data[i]["label"]
    totalDict["label_des"] = data[i]["label_des"]
    title_content = data[i]["sentence"]
    tempList = get_split(title_content, maxlen=120, overlap=40)
    totalDict["texts"] = tempList
    l1 = json.dumps(totalDict, ensure_ascii=False)
    fw.write(l1)
    fw.write('\n')
fw.close()