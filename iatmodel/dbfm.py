import torch.nn as nn
from .network import Net

#Dual Branch Fusion Module.
class CALayer(nn.Module):
    """
    Channel Attention Layer
    parameter: in_channel
    More detail refer to:
    """
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class Conv_LReLU_2(nn.Module):
    """
    Network component
    (Conv + LeakyReLU)
    """
    def __init__(self, in_channel, out_channel):
        super(Conv_LReLU_2, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.Conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.lrelu(self.Conv1(x))
        # x = self.lrelu(self.Conv2(x))
        return x
class DBFM_Module(nn.Module):
    """
    This is Dual Branch Fusion Module.
    Input: low light image and lighten image
    Output: RGB
    """

    def __init__(self, ):
        super(DBFM_Module, self).__init__()
        self.up2 = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.mono_conv1 = Conv_LReLU_2(3, 32)
        self.mono_conv2 = Conv_LReLU_2(32, 64)
        self.mono_conv3 = Conv_LReLU_2(64, 128)
        self.mono_conv4 = Conv_LReLU_2(128, 256)
        self.mono_up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.color_conv1 = Conv_LReLU_2(3, 32)
        self.color_conv2 = Conv_LReLU_2(32, 64)
        self.color_conv3 = Conv_LReLU_2(64, 128)
        self.color_conv4 = Conv_LReLU_2(128, 256)
        self.color_up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.dual_conv6 = Conv_LReLU_2(512, 256)
        self.dual_up6 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.dual_conv7 = Conv_LReLU_2(192, 64)
        self.dual_up7 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dual_conv8 = Conv_LReLU_2(96, 16)

        self.DBLE_out = nn.Conv2d(16, 3, kernel_size=1, stride=1)

        self.channel_attention6 = CALayer(256 * 2)
        self.channel_attention7 = CALayer(192)
        self.channel_attention8 = CALayer(96)
        # 添加池化灰度图层
        self.pool_g = nn.MaxPool2d(kernel_size=2)

    def forward(self, color, mono):
        # 求灰度图
        x_gray = []
        i = color.shape[0]
        for j in range(i):
            r, g, b = color[j][0, :, :], color[j][1, :, :], color[j][2, :, :]
            x_gray.append((1. - (0.299 * r + 0.587 * g + 0.114 * b)).unsqueeze(0))  # 灰度图

        x1_gray_img = torch.stack(x_gray)
        x2_gray = self.pool_g(x1_gray_img)
        x3_gray = self.pool_g(x2_gray)


        color_conv1 = self.color_conv1(color)
        color_pool1 = self.pool1(color_conv1)

        color_conv2 = self.color_conv2(color_pool1)
        color_pool2 = self.pool1(color_conv2)

        color_conv3 = self.color_conv3(color_pool2)
        color_pool3 = self.pool1(color_conv3)

        color_conv4 = self.color_conv4(color_pool3)
        color_up5 = self.color_up4(color_conv4)

        mono_conv1 = self.mono_conv1(mono)
        mono_pool1 = self.pool1(mono_conv1)

        mono_conv2 = self.mono_conv2(mono_pool1)
        mono_pool2 = self.pool1(mono_conv2)

        mono_conv3 = self.mono_conv3(mono_pool2)
        mono_pool3 = self.pool1(mono_conv3)

        mono_conv4 = self.mono_conv4(mono_pool3)
        mono_up5 = self.mono_up4(mono_conv4)

        color_conv3 = color_conv3 * x3_gray

        concat6 = torch.cat([color_up5, mono_up5, color_conv3, mono_conv3], 1)
        ca6 = self.channel_attention6(concat6)
        dual_conv6 = self.dual_conv6(ca6)
        dual_up6 = self.dual_up6(dual_conv6)

        color_conv2 = color_conv2 * x2_gray

        concat7 = torch.cat([dual_up6, color_conv2, mono_conv2], 1)
        ca7 = self.channel_attention7(concat7)
        dual_conv7 = self.dual_conv7(ca7)
        dual_up7 = self.dual_up7(dual_conv7)

        color_conv1 = color_conv1 * x1_gray_img

        concat8 = torch.cat([dual_up7, color_conv1, mono_conv1], 1)
        ca8 = self.channel_attention8(concat8)
        dual_conv8 = self.dual_conv8(ca8)

        DBLE_out = self.lrelu(self.DBLE_out(dual_conv8))

        return DBLE_out


class DB_iat(nn.Module):
    def __init__(self):
        super(DB_iat, self).__init__()
        self.iat = Net()
        self.db = DBFM_Module()

    def forward(self, x):
        out1 = self.iat(x)
        # torchvision.utils.save_image(out1, "testiat.png")
        out = self.db(x,out1)
        return out


from torch import nn


from lib.utils import e_gray,aMask
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
        self.fc_input_size = None
        self.gamma_base1 = nn.Parameter(torch.ones((1)), requires_grad=True)
        self.gamma_base2 = nn.Parameter(torch.ones((1)), requires_grad=True)
        self.base_1 = nn.Parameter(torch.ones(8), requires_grad=True)
    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 1) #二维 HW*C
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])  # 指定维度做内积
        image = image.view(shape)#变回原来维度
        return torch.clamp(image, 1e-8, 1.0)#张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量。

    def forward(self, img):
        # torchvision.utils.save_image(I, "1.png")
        I = e_gray(img)
        # 自适应Mask生成
        I_Mask = aMask(I)
        img_g = torch.cat((img,I), dim=1)
        feature = self.conv(img_g)
        if self.fc_input_size is None:
            self.fc_input_size = feature.shape[1] * feature.shape[2] * feature.shape[3]
            self.fc[0] = nn.Linear(self.fc_input_size, 120)
            self.fc[0].cuda()
        # x =feature.view(img.shape[0], -1)
        out = self.fc(feature.view(img.shape[0], -1))
        g1, g2, c = out[:, 0].unsqueeze(1), out[:, 1].unsqueeze(1), out[:, 2:]
        g1 = g1 + self.gamma_base1
        g2 = g2 + self.gamma_base2
        c = c + self.base_1

        b = img.shape[0]
        img_high = torch.stack([self.apply_color(img[k, :, :, :], c[k, :]) for k in range(b)], dim=0) #c
        output = torch.where(I_Mask == 0, img_high ** g1[:, :, None, None], img_high ** g2[:, :, None, None])

        return output

# 通道混洗
def channel_shuffle(x, groups):

    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # num_channels = groups * channels_per_group

    # grouping, 通道分组
    # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()
    # x.shape=(batchsize, channels_per_group, groups, height, width)
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

import os
import torch

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    f_u= DB_iat().cuda()
    print(f_u)
    # 计算参数量
    print('total parameters:', sum(param.numel() for param in f_u.parameters()))
    # high = net(low_image)