import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16, vgg19
from pytorch_msssim import ms_ssim, ssim
import numpy as np
import cv2

def SSIM(out_image, gt_image):
    out_image = torch.clamp(out_image, min=0.0, max=1.0)
    return ssim(out_image, gt_image, data_range=1, size_average=True)
def PSNR(out_image, gt_image):
    out_image = torch.clamp(out_image, min=0.0, max=1.0)
    mse = torch.mean((out_image/1.0 - gt_image/1.0)**2)
    return 10 * torch.log10(1.0 / mse)
def MS_SSIMLoss(out_image, gt_image):
    return 1 - ms_ssim(out_image, gt_image, data_range=1, size_average=True)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class Colorloss(nn.Module):
    def __init__(self):
        super(Colorloss, self).__init__()
        # self.Colorloss_weight = Colorloss_weight

    def forward(self, high_img,enhance_img):
        # batch_size = x.size()[0]
        # color loss
        b, c, h, w = high_img.shape
        true_reflect_view = high_img.view(b, c, h * w).permute(0, 2, 1)
        pred_reflect_view = enhance_img.view(b, c, h * w).permute(0, 2, 1)  # 16 x (512x512) x 3
        true_reflect_norm = torch.nn.functional.normalize(true_reflect_view, dim=-1)
        pred_reflect_norm = torch.nn.functional.normalize(pred_reflect_view, dim=-1)
        cose_value = true_reflect_norm * pred_reflect_norm
        cose_value = torch.sum(cose_value, dim=-1)  # 16 x (512x512)  # print(cose_value.min(), cose_value.max())
        color_loss = torch.mean(1 - cose_value)
        return color_loss

# Color Loss
class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k

# --- Perceptual loss network  --- #
class PerceptualLoss(nn.Module):

    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg16(pretrained=True).cuda()
        # vgg.load_state_dict(torch.load("D:\study\code\lab\model/vgg16-397923af.pth"))
        self.loss_network = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False

        self.l1_loss = nn.L1Loss()

    def normalize_batch(self, batch):
        # Normalize batch using ImageNet mean and std
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return (batch - mean) / std

    def forward(self, out_images, target_images):

        loss = self.l1_loss(
            self.loss_network(self.normalize_batch(out_images)),
            self.loss_network(self.normalize_batch(target_images))
        )

        return loss

# Perpectual Loss
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)

def e_gray(X):
    # 求光照注意力图
    in_gray_img = []
    i = X.shape[0]
    for j in range (i) :
        r, g, b = X[j][0, :, :], X[j][ 1, :, :], X[j][2, :, :]
        in_gray_img.append( (1.0- (0.299 * r + 0.587 * g + 0.114 * b)).unsqueeze(0))  # 关注较暗的部分
    in_gray_img_ten = torch.stack(in_gray_img)
    return in_gray_img_ten

# import matplotlib.pyplot as plt
# #绘制直方图
# def plot_hist(hist):
#     hist_np = hist.cpu().numpy()
#     # 创建x轴坐标
#     x = range(len(hist_np))
#     # 绘制直方图
#     plt.bar(x, hist_np)
#     plt.show()

def aMask(images):
    # 计算灰度图阈值，取阈值的百分之八十
    # 获取批量大小
    batch_size = images.shape[0]
    # 初始化掩码
    masks = torch.zeros_like(images)
    # 处理每个图像
    for i in range(batch_size):
        image = images[i]
        # 计算直方图
        hist = torch.histc(image, bins=256, min=0, max=1)
        # 计算总像素数
        total_pixels = image.numel()
        # 计算阈值
        cum_hist = torch.cumsum(hist, dim=0)
        threshold = torch.searchsorted(cum_hist, 0.8 * total_pixels).item() / 256
        # threshold = torch.argmax(hist) / 1.25
        # 创建掩码
        mask = torch.zeros_like(image)
        mask[image >= threshold] = 1
        masks[i] = mask
    return masks

    # # 计算直方图
    # hist = torch.histc(image, bins=256, min=0, max=1)
    # plot_hist(hist)
    # 显示直方图
    # plt.hist(img.ravel(), 256, [0, 256])
    # plt.show()
    # # # 计算阈值
    # # threshold = torch.argmax(hist) / 1.25
    # # 计算总像素数
    # total_pixels = image.numel()
    # # 计算阈值
    # cum_hist = torch.cumsum(hist, dim=0)
    # threshold = torch.searchsorted(cum_hist, 0.8 * total_pixels).item() / 256
    # # 打印阈值
    # print(f'Threshold: {threshold}')
    # # 创建掩码
    # mask = torch.zeros_like(image)
    # mask[image >= threshold] = 1
    # return mask

def e_illumination_SSR(X):
    # X: input image with shape [B, C, H, W]
    # I: estimated illumination map with shape [B, 1, H, W]

    # Convert image to grayscale
    X_gray = torch.mean(X, dim=1, keepdim=True)

    # Apply Gaussian blur to grayscale image
    kernel_size = 5
    sigma = 3
    kernel = torch.zeros(1, 1, kernel_size, kernel_size)
    kernel[0, 0] = torch.tensor([[1.0, 4.0, 6.0, 4.0, 1.0],
                                 [4.0, 16.0, 24.0, 16.0, 4.0],
                                 [6.0, 24.0, 36.0, 24.0, 6.0],
                                 [4.0, 16.0, 24.0, 16.0, 4.0],
                                 [1.0, 4.0, 6.0, 4.0, 1.0]]) / 256
    I = F.conv2d(X_gray.log(), kernel.cuda(),padding=2)

    return I.exp()

def e_illumination_MSR(X):
    # X: input image with shape [B, C, H, W]
    # I: estimated illumination map with shape [B, 1, H, W]

    # Convert image to grayscale
    X_gray = torch.mean(X, dim=1, keepdim=True)

    # Define scales for Gaussian blur
    scales = [15, 80, 250]

    # Apply Gaussian blur to grayscale image at multiple scales
    I = 0
    for scale in scales:
        kernel_size = scale // 10 * 2 + 1
        sigma = scale / 10
        kernel = torch.zeros(1, 1, kernel_size, kernel_size)
        kernel[0, 0] = torch.exp(
            -torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float().pow(2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        # kernel = kernel.permute(2, 3, 1, 0)
        I += F.conv2d(X_gray.log(), kernel.cuda(),padding=1)

    I /= len(scales)

    return I.exp()
# import os
# from PIL import Image
# if __name__ == "__main__":
#     # -----测试网络结构
#     os.environ['CUDA_VISIBLE_DEVICES']='0'
#     data = Image.open("D:\study\code\datasets\LOL\eval15\High/1.png")
#
#     low_image = torch.Tensor(low_image)
#     I = e_gray(low_image)
#     mask = aMask(I)
