import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# 设置随机种子
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Clip(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.encode_image
        self.text_encoder = clip_model.encode_text
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, text):
        # image = self.preprocess(image).unsqueeze(0).to(device)
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load("RN50", device="cuda")

    # 加载 Caltech101 数据集
    dataset = ImageFolder(root='./caltech101/101_ObjectCategories', transform=preprocess)

    # 初始化CoOp模型、损失函数和优化器
    model = Clip(clip_model).to(device)

    # 计算数据集大小
    dataset_size = len(dataset)
    # 计算划分的样本数量
    test_size = int(0.1 * dataset_size)
    test_set, _ = random_split(dataset, [test_size, dataset_size - test_size])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)

    # 获取模型文字输入
    classnames = [name.replace("_", " ") for name in dataset.classes]
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classnames]).to(device)

    # 评估模型
    model.eval()
    # 计算测试精度
    with torch.no_grad():
        correct_test = 0
        total_test = 0
        # print(len(test_loader))
        for images_test, labels_test in test_loader:
            images_test = images_test.to(device)
            labels_test = labels_test.to(device)
            outputs_test = model(images_test, text_inputs)
            _, predicted_test = torch.max(outputs_test.data, 1)
            total_test += labels_test.size(0)
            correct_test += (predicted_test == labels_test).sum().item()
            test_acc = correct_test / total_test
            print(test_acc, correct_test, total_test)
        print('测试集测试精度：{:.4f}'.format(test_acc))
