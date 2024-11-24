import os
import cv2
from PIL import Image,ImageFile
import numpy as np
import torchvision.transforms as transforms
import torch.optim
import glob
import random
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Normalize, ConvertImageDtype
# ImageFile.LOAD_TRUNCATED_IMAGES = True #错误文件跳过

def populate_train_list(limages_path, mode='train'):
    # print(images_path)
    # image_list_lowlight = glob.glob(limages_path + '/*.png')#LOL,lcdp,NPE
    image_list_lowlight = glob.glob(limages_path + '/*.*')#5k,DICM,VV
    # image_list_lowlight = glob.glob(limages_path + '/*.JPG')
    # image_list_lowlight = glob.glob(limages_path + '/*.bmp')#LIME
    train_list = image_list_lowlight

    if mode == 'train':
        random.shuffle(train_list)

    return train_list

#dataloader
class lowlight_loader(data.Dataset):

    def __init__(self, images_path, mode='train', normalize=True,size_correct=True):
        self.train_list = populate_train_list(images_path, mode)
        # self.h, self.w = int(img_size[0]), int(img_size[1])
        # train or test
        self.mode = mode
        self.data_list = self.train_list
        self.normalize = normalize
        # self.size_correct = False
        self.size_correct = size_correct
        print("Total examples:", len(self.train_list))

    # Data Augmentation
    # TODO: more data augmentation methods
    # def FLIP_LR(self, low, high):
    #     if random.random() > 0.5:
    #         low = low.transpose(Image.FLIP_LEFT_RIGHT)  # 转置？
    #         high = high.transpose(Image.FLIP_LEFT_RIGHT)
    #     return low, high
    #
    # def FLIP_UD(self, low, high):
    #     if random.random() > 0.5:
    #         low = low.transpose(Image.FLIP_TOP_BOTTOM)
    #         high = high.transpose(Image.FLIP_TOP_BOTTOM)
    #     return low, high
    #
    # def get_params(self, low):
    #     self.w, self.h = low.size
    #
    #     self.crop_height = random.randint(self.h / 2, self.h)  # random.randint(self.MinCropHeight, self.MaxCropHeight)
    #     self.crop_width = random.randint(self.w / 2, self.w)  # random.randint(self.MinCropWidth,self.MaxCropWidth)
    #     # self.crop_height = 224 #random.randint(self.MinCropHeight, self.MaxCropHeight)
    #     # self.crop_width = 224 #random.randint(self.MinCropWidth,self.MaxCropWidth)
    #
    #     i = random.randint(0, self.h - self.crop_height)
    #     j = random.randint(0, self.w - self.crop_width)
    #     return i, j
    #
    # def Random_Crop(self, low, high):
    #     self.i, self.j = self.get_params((low))
    #     if random.random() > 0.5:
    #         low = low.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
    #         high = high.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
    #     return low, high

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]

        if self.mode == 'train':
            # 用cv2打开图片
            # data_lowlight = cv2.imread(data_lowlight_path)  # 打开为RGB格式
            # data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_BGR2RGB)
            # data_highlight = cv2.imread(data_lowlight_path.replace('Low', 'High'))
            # data_highlight = cv2.imread(data_lowlight_path.replace('low0', 'normal0').replace('Low', 'Normal'))  # LOLV2
            # data_highlight = cv2.cvtColor(data_highlight, cv2.COLOR_BGR2RGB)
            # data_hsv = cv2.cvtColor(data_lowlight, cv2.COLOR_BGR2HSV)
            #使用PIL
            data_lowlight = Image.open(data_lowlight_path)  # 打开为RGB格式
            # data_highlight = Image.open(data_lowlight_path.replace('Low', 'High').replace('low0', 'high0'))#LOL
            # data_highlight = Image.open(data_lowlight_path.replace('input', 'expertC_gt'))#5k
            data_highlight = Image.open(data_lowlight_path.replace('input', 'gt'))  #lcdp dataset
            # ======== ME 数据集读取
            # filename= data_lowlight_path.replace('INPUT_IMAGES', 'GT_IMAGES').split('-')
            # data_highlight = Image.open(filename[0]+"-"+filename[1]+".jpg")
            #======== SID 数据集读取
            # filename= data_lowlight_path.replace('inputimage', 'gtimage').split("_00_")
            # data_highlight = Image.open(filename[0]+".png")
            # data_highlight = Image.open(data_lowlight_path.replace('low','normal').replace('Low','Normal').replace('input','expertC_gt'))

            # data_lowlight, data_highlight = self.FLIP_LR(data_lowlight, data_highlight)
            # data_lowlight, data_highlight = self.FLIP_UD(data_lowlight, data_highlight)
            # data_lowlight, data_highlight = self.Random_Crop(data_lowlight, data_highlight)

            # print(self.w, self.h)
            # print(data_lowlight.size, data_highlight.size)
            #图像尺寸调整
            if self.size_correct == True:
                # data_lowlight = cv2.resize(data_lowlight,(512,512))
                # data_highlight = cv2.resize(data_highlight, (512, 512))
                # data_hsv = cv2.resize(data_hsv, (512, 512))
                data_lowlight = data_lowlight.resize((512,512), Image.ANTIALIAS)
                data_highlight = data_highlight.resize((512,512), Image.ANTIALIAS)
                # data_hsv = data_hsv.resize((512,512),Image.ANTIALIAS)
            data_lowlight, data_highlight = (np.asarray(data_lowlight) / 255.0), (np.asarray(data_highlight) / 255.0)

            #正则化
            if self.normalize:
                # data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(data_highlight).float()
                transform_input = Compose(
                    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ConvertImageDtype(torch.float), ])
                transform_gt = Compose([ToTensor(), ConvertImageDtype(torch.float), ])
                # return transform_input(data_lowlight).permute(2, 0, 1), transform_gt(data_highlight).permute(2, 0, 1)
                return transform_input(data_lowlight), transform_gt(data_highlight)
            else:
                data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(data_highlight).float()
                return data_lowlight.permute(2, 0, 1), data_highlight.permute(2, 0, 1)

        elif self.mode == 'test':
            # 用cv2打开图片
            # data_lowlight = cv2.imread(data_lowlight_path)  # 打开为BGR格式
            # data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_BGR2RGB)#转换 rmal0').replace('Low','Normal'))#LOLV2
            # data_highlight = cv2.cvtColor(data_highlight, cv2.COLOR_BGR2RGB)
            # data_hsv = cv2.cvtColor(data_lowlight, cv2.COLOR_BGR2HSV)

            data_lowlight = Image.open(data_lowlight_path)
            data_highlight = Image.open(data_lowlight_path.replace('Low', 'High').replace('low', 'high'))  # LOL
            # data_highlight = Image.open(data_lowlight_path.replace('low0', 'normal0').replace('Low','Normal')) #LOLv2
            # data_highlight = Image.open(data_lowlight_path.replace('test', 'test_gt'))#5k_512pix
            # data_highlight = Image.open(data_lowlight_path.replace('test-input', 'test-gt'))  #lcdp
            # ========ME数据集读取=========
            # filename = data_lowlight_path.replace('INPUT_IMAGES', 'expert_c_testing_set').split("-")# ME
            # data_highlight = Image.open(filename[0] + ".jpg") # ME

            if self.size_correct == True:
                data_lowlight = data_lowlight.resize((512,512), Image.ANTIALIAS)
                data_highlight = data_highlight.resize((512,512), Image.ANTIALIAS)
            data_lowlight, data_highlight = (np.asarray(data_lowlight) / 255.0), (np.asarray(data_highlight) / 255.0)

            # 将图像rgb转换成Tensor
            if self.normalize:
                transform_input = Compose(
                    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ConvertImageDtype(torch.float), ])
                transform_gt = Compose([ToTensor(), ConvertImageDtype(torch.float), ])
                print(transform_input(data_lowlight))
                print(transform_input(data_highlight))
                return transform_input(data_lowlight), transform_gt(data_highlight),data_lowlight_path
            else:
                data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(
                    data_highlight).float()
                return data_lowlight.permute(2, 0, 1), data_highlight.permute(2, 0, 1),data_lowlight_path

    def __len__(self):
        return len(self.data_list)


import argparse

def load_init(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser(description='PyTorch Low-Light Enhancement')
    parser.add_argument('--normalize', default=False)
    parser.add_argument('--resize', default=False)
    opt = parser.parse_args()
    print(opt)

    # Data Setting
    train_dataset = lowlight_loader(images_path=config.train_LL_folder, normalize=opt.normalize,mode="train",size_correct = config.resize)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1,
                                             pin_memory=True)
    val_dataset = lowlight_loader(images_path=config.img_val_path, normalize=opt.normalize,mode="test",size_correct = config.resize)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1,
                                               pin_memory=True)

    return train_loader,val_loader
def load_init_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser(description='PyTorch Low-Light Enhancement')
    parser.add_argument('--test_LL_folder', default="D:/study/code/datasets/LOL/eval15/Low")
    # parser.add_argument('--test_LL_folder', default="D:/study/code/datasets/LOL_v2/Test/Low")
    # parser.add_argument('--test_LL_folder', default="D:/study/code/datasets/lcdp_dataset/test-input")
    # parser.add_argument('--test_LL_folder', default="D:/study/code/datasets/ME/testing/INPUT_IMAGES")
    # parser.add_argument('--test_LL_folder', default="D:/study/code/datasets/DICM")
    # parser.add_argument('--test_LL_folder', default="D:/study/code/datasets/LIME")
    # parser.add_argument('--test_LL_folder', default="D:/study/code/datasets/NPE")
    # parser.add_argument('--test_LL_folder', default="D:/study/code/datasets/VV")
    # parser.add_argument('--test_LL_folder', default="D:/study/code/datasets/fivek/test/")

    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--size_correct', default=False)# LOL
    # parser.add_argument('--size_correct', default=True)# others
    parser.add_argument('--normalize', default=False)

    config = parser.parse_args()

    # Data Setting
    test_dataset = lowlight_loader(images_path=config.test_LL_folder, normalize=config.normalize,mode='test',size_correct = config.size_correct)#测试返回的第三个参数为文件名
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1,
                                               pin_memory=True)
    return dataloader

if __name__ == '__main__':
    # load_init()
    #======测试读图像两种方法的差异
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser(description='PyTorch Low-Light Enhancement')
    parser.add_argument('--test_NL_folder', default="../datasets/fivek/expertC_gt_test")
    opt = parser.parse_args()
    # test_img(opt)
    # resize_pic(opt)


    #======测试灰度图/光照图读取