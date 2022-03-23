# -*- coding = utf-8 -*-
# @Time : 2022/3/17 20:50
# @Author : cxk
# @File : test.py
# @Software : PyCharm

import torch
from torchvision import transforms
import numpy as np
from torchvision.models.segmentation.segmentation import fcn_resnet50
import os
from PIL import Image
from matplotlib import pyplot as plt


if __name__ == '__main__':

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

    plt.subplot(1, 2, 1), plt.title("Predicted")
    plt.imshow(temp)

    gt = Image.open("./val/gt/Sp_D_NRN_R_arc0029_art0068_0568_gt.png")
    plt.subplot(1, 2, 2), plt.title("Ground Truth")
    plt.imshow(gt)

    plt.show()
