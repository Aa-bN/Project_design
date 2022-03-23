from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
import numpy as np


class CASIA(Dataset):

    def __init__(self, imgDir=None, gtDir=None, transform=None):
        self.imgDir = imgDir
        self.gtDir = gtDir
        self.imgNameList = os.listdir(self.imgDir)
        self.gtNameList = os.listdir(self.gtDir)
        self.transform = transform

    def __getitem__(self, idx):
        imgName = self.imgNameList[idx]
        imgPath = os.path.join(self.imgDir, imgName)
        gtName = self.gtNameList[idx]
        gtPath = os.path.join(self.gtDir, gtName)

        img_init = Image.open(imgPath)
        img = img_init.convert("RGB")
        gt_init = Image.open(gtPath)
        gt = gt_init.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
            gt = self.transform(gt)

        return img, gt

    def __len__(self):
        return len(self.imgNameList)

    def check(self):
        num_right = 0
        for i in range(0, len(self.imgNameList)):
            tempImgName = os.path.splitext(self.imgNameList[i])[0]
            tempGtName = os.path.splitext(self.gtNameList[i])[0]
            tempGtName = tempGtName[:-3]
            if tempImgName == tempGtName:
                num_right += 1
        print("num_right: ", num_right)
        if num_right == len(self.imgNameList):
            print("Image and ground truth are correct.")
        else:
            print("Image and ground truth do not match.")


def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=0)
    return batched_imgs, batched_targets


def cat_list(pictures, fill_value=0):
    # 计算该batch中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[pic.shape for pic in pictures]))
    batch_shape = (len(pictures),) + max_size
    batched_pics = pictures[0].new(*batch_shape).fill_(fill_value)
    for pic, pad_pic in zip(pictures, batched_pics):
        pad_pic[..., :pic.shape[-2], :pic.shape[-1]].copy_(pic)
    return batched_pics


if __name__ == '__main__':

    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    testdata = CASIA(imgDir="./train/image/", gtDir="./train/groundtruth/", transform=None)
    testdata.check()
    img, gt = testdata[3]
    plt.imshow(img)
    plt.show()
    plt.imshow(gt)
    plt.show()
    testdata2 = CASIA(imgDir="./train/image/", gtDir="./train/groundtruth/", transform=trans)

    dataloader2 = DataLoader(dataset=testdata2, batch_size=4, shuffle=False, collate_fn=collate_fn)
    for i, (img, gt) in enumerate(dataloader2):
        print(i)
        print(img.shape)
        print(img[0].shape)
        print(gt[0].shape)
        print(gt[0].max(dim=2))
        print(np.unique(gt[0]))
        print(np.unique(img[0]))
        if i == 1:
            img = img[0]
            gt = gt[0]
            toPIL = transforms.ToPILImage()
            img = toPIL(img)
            gt = toPIL(gt)
            img.save("lalala.png")
            gt.save("lalala_gt.png")
            plt.imshow(img)
            plt.show()
            plt.imshow(gt)
            plt.show()
        if i == 5:
            break
