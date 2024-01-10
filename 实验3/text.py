import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import jieba
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader
from tqdm import tqdm
import mymodels as mm


sentence_max_size = 50
batch_size = 64
num_epochs = 50
label_num = 10
lr = 0.1
hidden_size = 512

# 获取数据
data_file = os.path.join('.', 'online_shopping_10_cats.csv')
data = pd.read_csv(data_file, )
del data['label']
# 编号分类
cat_mapping = {'书籍': 0, '平板': 1, '手机': 2, '水果': 3, '洗发水': 4, '热水器': 5, '蒙牛': 6,
               '衣服': 7, '计算机': 8, '酒店': 9}
data['cat'] = data['cat'].map(cat_mapping)
data['review'] = data['review'].astype(str)
# 词汇量67968
stopwords = [line.strip() for line in open('hit_stopwords.txt', encoding='utf-8').readlines()]

for i, sentence in tqdm(enumerate(data['review']), total=len(data)):
    sen_spl = []
    words = jieba.cut(sentence)
    for word in words:
        if word not in stopwords:
            sen_spl.append(word)
    data.at[i, 'review'] = sen_spl

# 获取数据集总条目数
total_entries = len(data)

# 划分数据集
train_indices = []
val_indices = []
test_indices = []

for i in range(total_entries):
    if i % 5 == 4:
        val_indices.append(i)
    elif i % 5 == 0:
        test_indices.append(i)
    else:
        train_indices.append(i)

# 根据索引划分数据集
train_data = data.iloc[train_indices]
val_data = data.iloc[val_indices]
test_data = data.iloc[test_indices]
# 还原索引
train_data.reset_index(drop=True, inplace=True)
val_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

word2vec_model_path = './tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'  # 词向量文件的位置
# 加载词向量模型
wv = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=False, unicode_errors='ignore')
# wv = Word2Vec.load('word2vec.model')
# 创建nn.Embedding层
embedding_dim = wv.vector_size
# word2id是一个字典，存储{word:id}的映射
word2id = {word: idx for idx, word in enumerate(wv.index_to_key)}
# 根据已经训练好的词向量模型，生成Embedding对象
embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors))

# 数据集、数据加载器
train_set = mm.MyDataset(data=train_data, embedding=embedding, seq_len=sentence_max_size, word2id=word2id)
val_set = mm.MyDataset(data=val_data, embedding=embedding, seq_len=sentence_max_size, word2id=word2id)
test_set = mm.MyDataset(data=test_data, embedding=embedding, seq_len=sentence_max_size, word2id=word2id)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# 设置设备（使用 GPU 如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
model = mm.MyBiRNN(input_size=embedding_dim, hidden_size=hidden_size, output_size=10)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
print(f'开始训练，设备：{device}')

mm.train(model, train_loader, num_epochs, optimizer, criterion, device)

train_acc, train_re, train_f1 = mm.test(model, train_loader, device)
print('训练集测试精度：{:.4f}，召回率：{:.4f}，F1：{:.4f}'.format(train_acc, train_re, train_f1))
val_acc, val_re, val_f1 = mm.test(model, val_loader, device)
print('验证集测试精度：{:.4f}，召回率：{:.4f}，F1：{:.4f}'.format(val_acc, val_re, val_f1))
test_acc, test_re, test_f1 = mm.test(model, test_loader, device)
print('测试集测试精度：{:.4f}，召回率：{:.4f}，F1：{:.4f}'.format(test_acc, test_re, test_f1))
