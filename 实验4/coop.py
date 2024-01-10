import random

import numpy as np
import timm.scheduler
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


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initialization
        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.to(device)).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts


class CoOp(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.encode_image
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        # image = self.preprocess(image).unsqueeze(0).to(device)
        image_features = self.image_encoder(image)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        # print(loss.item())
        total_loss += loss.item()

    epoch_loss = total_loss / len(train_loader)
    return epoch_loss


# def collate_fn(batch):
#     return {
#         'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
#         'labels': torch.tensor([x['labels'] for x in batch])
#     }


if __name__ == '__main__':

    # 设置超参数
    num_embeddings = 16  # 可学习的文本嵌入维度
    batch_size = 1
    num_epochs = 100
    learning_rate = 0.002

    # # 定义归一化的均值和标准差
    # mean = [0.5, 0.5, 0.5]  # RGB三个通道的均值
    # std = [0.5, 0.5, 0.5]  # RGB三个通道的标准差
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)  # 归一化
    # ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load("RN50", device="cuda")

    # 加载 Caltech101 数据集
    dataset = ImageFolder(root='./caltech101/101_ObjectCategories', transform=preprocess)

    # 获取类别数量
    num_classes = len(dataset.classes)
    n = 2  # 每个类别选择的样本数量
    # 创建一个列表，用于存储每个类别的样本索引
    sample_indices = []

    # 遍历每个类别获取n个样本，实现few-shot
    for class_idx in range(num_classes):
        class_indices = np.zeros_like(dataset.targets)
        for i in range(len(class_indices)):
            if dataset.targets[i] == class_idx:  # 获取当前类别的样本索引
                class_indices[i] = 1
        class_indices = torch.tensor(class_indices)
        indices = torch.nonzero(class_indices).squeeze(1).tolist()  # 转换为列表形式的索引
        selected_indices = random.sample(indices, n)  # 随机选择 n 个样本索引
        sample_indices.extend(selected_indices)  # 添加到样本索引列表中

    # 创建一个SubsetRandomSampler，用于从指定的样本索引中进行采样
    sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_indices)

    # 创建DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    # print(len(train_loader))

    # 初始化CoOp模型、损失函数和优化器
    model = CoOp(dataset.classes, clip_model).to(device)

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # 定义学习率调度器
    scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer,
                                                 t_initial=num_epochs,
                                                 lr_min=1e-5,
                                                 warmup_t=1,
                                                 warmup_lr_init=1e-5)

    # 开始训练
    for epoch in range(num_epochs):
        scheduler.step(epoch)
        epoch_loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # # 保存模型参数
    # torch.save(model.state_dict(), './trained/2shot42.pth')

    # 计算数据集大小
    dataset_size = len(dataset)
    # 计算划分的样本数量
    test_size = int(0.1 * dataset_size)
    test_set, _ = random_split(dataset, [test_size, dataset_size - test_size])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    # 评估模型
    model.eval()
    # 计算测试精度
    with torch.no_grad():
        correct_test = 0
        total_test = 0
        print(len(test_loader))
        for images_test, labels_test in test_loader:
            images_test = images_test.to(device)
            labels_test = labels_test.to(device)
            outputs_test = model(images_test)
            _, predicted_test = torch.max(outputs_test.data, 1)
            total_test += labels_test.size(0)
            correct_test += (predicted_test == labels_test).sum().item()
            test_acc = correct_test / total_test
            # print(test_acc, correct_test, total_test)
        print('测试集测试精度：{:.4f}'.format(test_acc))
