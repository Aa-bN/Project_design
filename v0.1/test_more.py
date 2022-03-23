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

    imgName = os.listdir("./val/images/")
    gtName = os.listdir("./val/gt/")

    plt.figure(figsize=(10, 10))

    for i in range(len(imgName)):
        imgPath = os.path.join("./val/images/", imgName[i])
        gtPath = os.path.join("./val/gt/", gtName[i])

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


def test_softmax():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    imgPath = r"./val/images/Sp_D_NRN_R_arc0029_art0068_0568.jpg"
    img_init = Image.open(imgPath)
    img = img_init.convert("RGB")
    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    toPIL = transforms.ToPILImage()

    img = trans(img)

    img = img.unsqueeze(0).to(device)
    model = torch.load("./model/fcnNet_20_nn_weight_lr=1e-1.pth").to(device)
    # model = fcn_resnet50(pretrained=False, progress=False, num_classes=2, aux_loss=False)
    # model.load_state_dict(torch.load("./model/fcnNet_para_14.pth"))
    model.to(device)
    out = model(img)['out']

    temp = out.squeeze(0)
    temp = torch.argmax(temp, dim=0).detach().cpu()

    pic = temp.float()
    pic = toPIL(pic)
    pic.save("testPic.png")

    # -------- with probability ----------
    temp2 = out.squeeze(0)
    temp2 = torch.softmax(temp2, dim=0)


    plt.subplot(1, 3, 1), plt.title("Predicted")
    plt.imshow(temp)

    gt = Image.open("./val/gt/Sp_D_NRN_R_arc0029_art0068_0568_gt.png")
    plt.subplot(1, 3, 3), plt.title("Ground Truth")
    plt.imshow(gt)

    plt.show()
    pass


if __name__ == '__main__':
    model_path = os.listdir("./model/")
    for i in range(len(model_path)):
        mPath = os.path.join("./model/", model_path[i])
        test_more(mPath)





