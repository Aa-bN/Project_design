# -*- coding = utf-8 -*-
# @Time : 2022/3/11 16:48
# @Author : cxk
# @File : train.py
# @Software : PyCharm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from torchvision.models.segmentation.segmentation import fcn_resnet50
from PIL import Image
from matplotlib import pyplot as plt
from dataSet import CASIA, collate_fn
import os

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 定义数据集及相关预处理操作
trans = transforms.Compose([
    transforms.ToTensor(),
])
train_data = CASIA(imgDir="./train/image/", gtDir="./train/groundtruth/", transform=trans)
train_data.check()
val_data = CASIA(imgDir="./val/image/", gtDir="./val/groundtruth", transform=trans)
val_data.check()
print("Dataset ready.")

# 加载数据
train_dataloader = DataLoader(dataset=train_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
val_dataloader = DataLoader(dataset=val_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
print("DataLoader ready.")

# 创建模型
fcnNet = fcn_resnet50(pretrained=False, progress=False, num_classes=2, aux_loss=False)
fcnNet = fcnNet.to(device)
print("Model ready.")

# 损失函数，其中的weight和reduction待定
weight = torch.Tensor([1, 16])
loss_func = nn.CrossEntropyLoss(reduction='mean', weight=weight)
loss_func = loss_func.to(device)

# 学习率和优化器
learning_rate = 1e-1
optimizer = torch.optim.SGD(fcnNet.parameters(), lr=learning_rate)

# 训练轮次
epoch = 20


if __name__ == '__main__':

    if not os.path.exists("./model/"):
        os.mkdir("./model/")

    total_train_step = 0
    total_val_step = 0
    min_loss_per_epoch = 0  # for validation
    a = {'min_loss_epoch': 0}                  # record the best model's epoch

    for k in range(epoch):
        print("Epoch {} Started...".format(k))

        print("Training...")
        # train
        for i, (img, gt) in enumerate(train_dataloader):
            # 梯度清零
            optimizer.zero_grad()

            # for CASIA's gt, [batch, C=3/C=1, H, W] -> [batch, H, W]
            gt = gt[:, 0, :, :]

            img = img.to(device)
            gt = gt.to(device, dtype=torch.int64)

            outputs = fcnNet(img)['out']

            # 计算损失
            loss = loss_func(outputs, gt)

            # 反向传播
            loss.backward()

            # 梯度更新
            optimizer.step()

            total_train_step += 1

            if total_train_step % 100 == 0:
                print("Train_Epoch:{}, Step:{}, single_loss:{}".format(k, total_train_step, loss.item()))

        print("Validating...")
        # val
        val_loss_per_epoch = 0
        with torch.no_grad():
            for i, (img, gt) in enumerate(val_dataloader):
                gt = gt[:, 0, :, :]
                img = img.to(device)
                gt = gt.to(device, dtype=torch.int64)

                outputs = fcnNet(img)['out']
                loss = loss_func(outputs, gt)
                val_loss_per_epoch = val_loss_per_epoch + loss.item()

                total_val_step += 1

                if total_val_step % 50 == 0:
                    print("Val_Epoch:{}, Step:{}, single_loss:{}".format(k, total_val_step, loss.item()))

        if epoch == 0:
            min_loss_per_epoch = val_loss_per_epoch

        if val_loss_per_epoch < min_loss_per_epoch:
            min_loss_per_epoch = val_loss_per_epoch
            a["min_loss_epoch"] = k
            torch.save(fcnNet, './model/fcnNet_best.pth')

        print("Epoch {} Over. Min_loss is {}, from Epoch {}.".format(k, min_loss_per_epoch, a['min_loss_epoch']))

        if k == epoch-1:
            torch.save(fcnNet, "./model/fcnNet_last.pth")
        # torch.save(fcnNet.state_dict(), "./model/fcnNet_para_{}.pth".format(k))
    print("------")











