from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
import numpy as np


class CASIA(Dataset):

    def __init__(self, imgDir=None, gtDir=None, use="PIL", transform=None):
        self.imgDir = imgDir
        self.gtDir = gtDir
        self.imgNameList = os.listdir(self.imgDir)
        self.gtNameList = os.listdir(self.gtDir)
        self.use = use
        self.transform = transform

    def __getitem__(self, idx):
        imgName = self.imgNameList[idx]
        imgPath = os.path.join(self.imgDir, imgName)
        gtName = self.gtNameList[idx]
        gtPath = os.path.join(self.gtDir, gtName)

        # if self.use == 'cv2':
        #     # openCV默认过滤A通道
        #     # 图像由默认的BGR转化为RGB格式
        #     img_BGR = cv2.imread(imgPath)
        #     img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        #     gt_BGR = cv2.imread(gtPath)
        #     gt = cv2.cvtColor(gt_BGR, cv2.COLOR_BGR2RGB)

        # if self.use == 'PIL':
        # cv2 打开的图片和 torchvision.transform 预处理图片时，不首先进行ToTensor的话，可能产生冲突
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

    # 使用ToTensor，最终dataloader中图片对应的shape为 [N, C, H, W]
    # 否则仍然保持原始RBG图片 [H, W, C]的形状，最终体现为[N, H, W, C]
    # trans = transforms.ToTensor()

    # testdata = CASIA(imgDir="./image/Sp", gtDir="./groundtruth/Sp/", use="cv2", transform=trans)
    # testdata.check()
    # exit(0)
    # testdata = CASIA(imgDir="./image/Sp", gtDir="./groundtruth/Sp/")
    # img_test, gt_test = testdata[4]
    # plt.imshow(img_test)
    # plt.show()
    # plt.imshow(gt_test)
    # plt.show()

    # dataloader 存在的问题
    # 默认情况下，将图片堆叠在一起，成为[N, C, H, W]的张量
    # 故要求图片的尺寸是相同的，否则会报错
    # 在不裁减原图片的情况下
    # 解决方案有二：一是将batch_size=1，二是自定义collate_fn函数
    # 方案三：在数据集同一组图片进行相同形式的cut（而不是resize）
    # 但第二种方法目前来看多此一举
    # -------------------------------------------------------------------------------------------------------------
    '''
    dataloader = DataLoader(dataset=testdata, batch_size=1, shuffle=False)
    for i, (img, gt) in enumerate(dataloader):
        print(i)
        print(type(img))
        print(img.shape)
        print(type(gt))
        print(img.shape)
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(gt)
        # plt.show()
        if i == 10:
            break
    '''
    # -------------------------------------------------------------------------------------------------------------
    '''
    def my_collate(batch):
        img = [item[0] for item in batch]
        gt = [item[1] for item in batch]
        return img, gt

    dataloader2 = DataLoader(dataset=testdata, batch_size=4, shuffle=False, collate_fn=my_collate)
    for i, (img, gt) in enumerate(dataloader2):
        print(i)
        print(img[0].shape)
        print(gt[0].shape)
        print(gt[0].max(dim=2))
        if i == 5:
            break
    '''
    # -------------------------------------------------------------------------------------------------------------
    # torchvision transforms相关
    '''
    testdata = CASIA(imgDir="./image/Sp", gtDir="./groundtruth/Sp/", use='PIL', transform=None)
    img, gt = testdata[4]
    plt.imshow(img)
    plt.show()
    plt.imshow(gt)
    plt.show()
    trans = transforms.Compose([
        transforms.CenterCrop((384, 384))
    ])
    img2 = trans(img)
    gt2 = trans(gt)
    plt.imshow(img2)
    plt.show()
    plt.imshow(gt2)
    plt.show()
    '''
    # ---
    '''
    trans = transforms.Compose([
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor()
    ])
    testdata3 = CASIA(imgDir="./image/Sp/", gtDir="./groundtruth/Sp/", use="PIL", transform=trans)
    dataloader3 = DataLoader(dataset=testdata3, batch_size=4, shuffle=False)
    for i, (img, gt) in enumerate(dataloader3):
        print(i)
        print(img.shape)
        print(img[0].shape)
        print(gt[0].shape)
        print(gt[0].max(dim=2))
        print(np.unique(gt[0]))
        if i == 5:
            break
    '''

    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    testdata4 = CASIA(imgDir="./image/Sp/", gtDir="./groundtruth/Sp/", use="PIL", transform=trans)
    # img, gt = testdata4[4]
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(gt)
    # plt.show()
    dataloader4 = DataLoader(dataset=testdata4, batch_size=4, shuffle=False, collate_fn=collate_fn)
    for i, (img, gt) in enumerate(dataloader4):
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
