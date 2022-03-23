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

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 定义数据集及相关预处理操作
trans = transforms.Compose([
    # transforms.CenterCrop((384, 384)),
    transforms.ToTensor(),
])
train_data = CASIA(imgDir="./image/Sp/", gtDir="./groundtruth/Sp/", use="PIL", transform=trans)
train_data.check()
print("Dataset ready.")

# 加载数据
train_dataloader = DataLoader(dataset=train_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
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


def a_test():
    imgPath = r"./image/Sp/Sp_D_CNN_A_art0024_ani0032_0268.jpg"
    img_init = Image.open(imgPath)
    img = img_init.convert("RGB")
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    img = trans(img)
    img = img.unsqueeze(0).to(device)
    out = fcnNet(img)['out']
    print(out.shape)

    toPIL = transforms.ToPILImage()
    out2 = out.squeeze().detach().cpu()     # .numpy()
    print(out2.shape)

    om = torch.argmax(out2, dim=0).detach().cpu()   # .numpy()

    # tensor CHW，需要float int64 -> float32
    om_pic = om.float()
    om_pic = toPIL(om_pic)
    om_pic.save("omHxW.png")

    print(om.shape)
    print(np.unique(om))
    plt.subplot(3, 3, 1), plt.title('om')
    plt.imshow(om), plt.axis('off')

    om1 = out2[0]
    plt.subplot(3, 3, 2), plt.title('out[0]')
    plt.imshow(om1), plt.axis('off')

    om2 = out2[1]
    plt.subplot(3, 3, 3), plt.title('out[1]')
    plt.imshow(om2), plt.axis('off')

    # -----------------------------------------------------
    om = om.unsqueeze(2)
    om_cat = torch.cat((om, om, om), dim=2)

    # # numpy HWC
    # om_pic2 = om_cat.numpy()
    # om_pic2 = toPIL(om_pic2)
    # om_pic2.save("omHxWxC.png")

    plt.subplot(3, 3, 4), plt.title('om_cat')
    plt.imshow(om_cat), plt.axis('off')

    om1 = om1.unsqueeze(2)
    om1_cat = torch.cat((om1, om1, om1), dim=2)
    plt.subplot(3, 3, 5), plt.title('om1_cat')
    plt.imshow(om1_cat), plt.axis('off')

    om2 = om2.unsqueeze(2)
    om2_cat = torch.cat((om2, om2, om2), dim=2)
    plt.subplot(3, 3, 6), plt.title('om2_cat')
    plt.imshow(om2_cat), plt.axis('off')

    # -----------------------------------------------------
    plt.subplot(3, 3, 7), plt.title('om')
    plt.imshow(om, cmap='gray'), plt.axis('off')

    plt.subplot(3, 3, 8), plt.title('out[0]')
    plt.imshow(om1, cmap='gray'), plt.axis('off')

    plt.subplot(3, 3, 9), plt.title('out[1]')
    plt.imshow(om2, cmap='gray'), plt.axis('off')

    plt.show()


if __name__ == '__main__':

    total_train_loss = 0
    total_train_step = 0
    min_loss_per_epoch = 10000
    a = {}

    for k in range(epoch):
        epoch_loss = 0
        print("Epoch {} Started...".format(k))
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
            total_train_loss = total_train_loss + loss
            epoch_loss = epoch_loss + loss

            print(" Step:{}, single_loss:{}".format(total_train_step, loss.item()))

        if epoch_loss < min_loss_per_epoch:
            min_loss_per_epoch = epoch_loss
            a["min_loss_epoch"] = k
            torch.save(fcnNet, './model/fcnNet_best.pth')

        print("Epoch {} Over. Loss of this epoc is {}.".format(k, epoch_loss.item()))
        print("Min_loss is {}, from Epoch {}".format(min_loss_per_epoch.item(), a['min_loss_epoch']))

        if k == epoch-1:
            torch.save(fcnNet, "./model/fcnNet_last.pth")
        # torch.save(fcnNet.state_dict(), "./model/fcnNet_para_{}.pth".format(k))
    print("Total loss:", total_train_loss)











