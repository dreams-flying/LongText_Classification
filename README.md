# LongText_Classification
长文本分类，解决长距离依赖问题。
采用Bert + overlap_split分段 + 门控卷积网络
# 实验环境
Python==3.6</br>
tensorflow==1.14.0</br>
keras==2.3.1</br>
bert4keras==0.10.9</br>
笔者使用了开源的bert4keras，一个keras版的transformer模型库。bert4keras的更多介绍参见[这里](https://github.com/bojone/bert4keras)。
# 项目目录
├── bert4keras</br>
├── data    存放数据</br>
├── pretrained_model    存放预训练模型</br>
├── weights    保存权重</br>
├── extract_model.py</br>
├── extract_vectorize.py    </br>
├── iflytek_clssification_train.py    可单独使用</br>
├── predict.py    评估和测试代码</br>
# 数据集
采用CLUE [IFLYTEK 长文本分类数据集](https://www.cluebenchmarks.com/introduce.html)，对原始数据进行了处理，处理好的数据存放在data/iflytek_public文件夹下。</br>
# 使用说明
1.下载[预训练语言模型](https://github.com/google-research/bert#pre-trained-models)</br>
  可采用BERT-Base, Chinese等模型</br>
  更多的预训练语言模型可参见[bert4keras](https://github.com/bojone/bert4keras)给出的权重。</br>
2.构建数据集(数据集已处理好)</br>
  进入data/iflytek_public文件夹下，解压得到train_data.json和dev_data.json</br>
3.运行```iflytek_clssification_train.py```，模型文件保存在weights/save_bert文件夹下。该文件可单独使用，可跳过这步直接进行下一步。</br>
4.运行```extract_vectorize.py```提取特征向量，得到train_data_extract.npy和dev_data_extract.npy</br>
5.训练分类模型</br>
```
extract_model.py
```
6.评估和测试
```
predict.py
```
# 结果
| 数据集 | accuracy |
| :------:| :------: |
| dev | 0.60562 |
| test | 0.6023 |
# Note
iflytek_clssification_train.py可单独使用，其本身就是一个分类模型，第4步可以直接用BERT提取特征向量；</br>
也可以利用训练数据先在bert进行finetune，得到新的模型文件，然后再进行第4步。
