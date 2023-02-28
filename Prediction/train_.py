import torch
import numpy as np
import pandas as pd
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.autograd import Variable
import time

pre = time.time()

filepath = "dataset/Gateway-preprocessed.txt"
with open(filepath, 'r') as f:
    dataset = f.read()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
device = 'cpu'

def split_dataset(dataset, split_ratio):
    shuffled_indices = np.random.permutation(len(dataset))
    train_indices = shuffled_indices[int(len(dataset) * split_ratio):]
    val_indices = shuffled_indices[:int(len(dataset) * split_ratio)]
    return dataset[train_indices], dataset[val_indices]

indexed_text = tokenizer.encode(dataset)
del(dataset)

dataset_cut = []
for i in range(len(indexed_text)//512):
    # 将字符串分段成长度为 512
    dataset_cut.append(indexed_text[i*512:i*512+512])
del(indexed_text)

dataset_tensor = torch.tensor(dataset_cut)

train_data, val_data = split_dataset(dataset_tensor,0.05)
print(train_data.shape,val_data.shape)




# 构建数据集和数据迭代器，设定 batch_size 大小为 2
train_set = TensorDataset(train_data,
                          train_data)  # 标签与样本数据相同
train_loader = DataLoader(dataset=train_set,
                          batch_size=2,
                          shuffle=True)
val_set = TensorDataset(val_data,val_data)
val_loader = DataLoader(dataset=val_set,
                          batch_size=1,
                          shuffle=False)




epoch = 30  # 循环学习 30 次

model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 定义优化器
print('1')

for i in range(epoch):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(
            target).to(device)

        optimizer.zero_grad()

        # (loss, logits, a) = model(data, labels=target)

        outputs = model(data, labels=target)
        loss = outputs[0]
        logits = outputs[1]

        total_loss += loss

        loss.backward()
        optimizer.step()

        if batch_idx == len(train_loader)-1:
            # 在每个 Epoch 的最后输出一下结果
            print('average loss:', total_loss/len(train_loader))
        print('2')
        
        with torch.no_grad():
            total_loss = 0
            total_samples = 0
            for inputs, target in val_loader:
                print('3')
                outputs = model(inputs, labels=target)
                loss = outputs[0]
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
            avg_loss = total_loss / total_samples
            print(avg_loss)

print('训练时间：', time.time()-pre)