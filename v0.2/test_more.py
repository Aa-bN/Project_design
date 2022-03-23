# -*- coding = utf-8 -*-
# @Time : 2022/3/22 18:48
# @Author : cxk
# @File : test_more.py
# @Software : PyCharm

import torch
from torchvision import transforms
import numpy as np
from torchvision.models.segmentation.segmentation import fcn_resnet50
import os
from PIL import Image
from matplotlib import pyplot as plt


def test_more(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    model = torch.load(model_path).to(device)

    imgName = os.listdir("./test/images/")
    gtName = os.listdir("./test/gt/")

    plt.figure(figsize=(10, 10))

    for i in range(len(imgName)):
        imgPath = os.path.join("./test/images/", imgName[i])
        gtPath = os.path.join("./test/gt/", gtName[i])

        img = Image.open(imgPath).convert("RGB")
        img = trans(img)
        img = img.unsqueeze(0).to(device)

        out = model(img)['out']

        temp = out.squeeze(0)
        temp = torch.argmax(temp, dim=0).detach().cpu()

        plt.subplot(len(imgName), 2, (i+1)*2-1), plt.title("Predicted", fontdict={'size': 10})
        plt.imshow(temp)

        gt = Image.open(gtPath)
        plt.subplot(len(imgName), 2, (i+1)*2), plt.title("Ground Truth", fontdict={'size': 10})
        plt.imshow(gt)

    plt.subplots_adjust(hspace=0.5)
    name = model_path + ".png"
    plt.savefig(name)
    plt.show()


if __name__ == '__main__':
    model_path = os.listdir("./model/")
    for i in range(len(model_path)):
        mPath = os.path.join("./model/", model_path[i])
        test_more(mPath)





