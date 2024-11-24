import os
import time
import cv2
import numpy
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from lib.utils import SSIM,PSNR

from skimage.metrics import mean_squared_error as mse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision

import lib.pytorch_ssim as pytorch_ssim
import argparse
import data_load
from iatmodel.dbfm import DB_iat
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Low-Light Enhancement')
parser.add_argument('--gpus', default=0, type=int, help='number of gpu')

opt = parser.parse_args()
print(opt)

# ssimcal = SSIM()

def checkpoint(model, epoch, opt):
    try:
        os.stat(opt.save_folder)
    except:
        os.mkdir(opt.save_folder)

    model_out_path = opt.save_folder + "{}_{}.pth".format(Name_Exp, epoch)
    torch.save(model.state_dict(), model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path

def eval_train(low,gt):
    #tensor转换为numpy数组
    low_img = low.cpu().detach().numpy()
    high_img = gt.cpu().detach().numpy()
    # print(type(low_img))
    sum_psnr = 0
    sum_ssim = 0
    sum_mse = 0
    # lpips = util_of_lpips.calc_lpips()


    for i in range(low_img.shape[0]):
        ll = low_img[i]
        hh = high_img[i]

        # (400,600,3)
        low_im = np.transpose(ll,(1,2,0))
        high_im = np.transpose(hh,(1,2,0))
        # test_ssim = ssim_val = compare_ssim(low_im, low_im, multichannel=True)

        #===psnr和ssim的计算
        psnr_val = compare_psnr(hh, ll) #(true,test)
        ssim_val = compare_ssim(low_im, high_im, multichannel=True)

        #评价指标:mse
        mse_val = mse(hh, ll)

        sum_psnr += psnr_val
        sum_ssim += ssim_val
        sum_mse += mse_val

        # print("---batch PSNR:{} SSIM:{} MSE:{} ".format(psnr_val, ssim_val,mse_val))

    avg_psnr = sum_psnr / len(low_img)
    avg_ssim = sum_ssim / len(low_img)
    avg_mse = sum_mse / len(low_img)

    return avg_psnr,avg_ssim,avg_mse
# from .iatmodel.utils import validation
def eval_test(model):
    # 读取数据集
    img = data_load.load_init_test()

    sum_psnr = 0
    sum_ssim = 0
    sum_niqe = 0
    sum_mse = 0
    sum_mae = 0
    mae_cal = nn.L1Loss()

    for iteration, imgs in tqdm(enumerate(img)):
        with torch.no_grad():
            low_img, high_img= imgs[0].cuda(), imgs[1].cuda()
            enhance_img = model(low_img)

            #保存图像结果
            # torchvision.utils.save_image(enhance_img, "./ablation/"+name[1])
            # ref = np.array(Image.open('./ablation/'+name[1]).convert('LA'))[:, :, 0]  # ref
            # niqe_val = niqe(ref)
            niqe_val = 0

            # print('test NIQE of ref parrot image is: %0.3f' % niqe(ref))
            ##每个batch的psnr，ssim均值
            psnr, ssim,mse = eval_train(enhance_img, high_img)
            mae_val = mae_cal(enhance_img,high_img)
            ssim = SSIM(enhance_img, high_img).item()
            psnr = PSNR(enhance_img, high_img).item()

            sum_psnr += psnr
            sum_ssim += ssim
            sum_niqe += niqe_val
            sum_mse += mse
            sum_mae += mae_val
            # print("PSNR :{} SSIM:{} NIQE:{} MSE:{}".format(psnr, ssim,niqe_val,mse))

    avg_psnr = sum_psnr / (iteration+1)
    avg_ssim = sum_ssim / (iteration+1)
    avg_niqe = sum_niqe / (iteration+1)
    avg_mse = sum_mse / (iteration+1)
    # avg_mae = sum_mae / (iteration+1)
    # print("AVG PSNR:{} AVG SSIM:{} AVG NIQR:{} AVG MSE:{}".format(avg_psnr,avg_ssim,avg_niqe,avg_mse))
    return avg_psnr,avg_ssim,avg_niqe,avg_mse


if __name__ == "__main__":
    #cfg()
    cuda = opt.gpus
    print("test cuda:",cuda)
    # =============================#
    #          Build model        #
    # =============================#
    print('===> Build model')
    model = DB_iat().cuda()
    # model = HD_iat().cuda()
    # model = FDB_iat().cuda()
    # model = LeNet().cuda()
    # model.load_state_dict(torch.load("./checkpoints/globe/golbe_100.pth", map_location=lambda storage, loc: storage), strict=True)
    print('---------- Networks architecture -------------')
    print(model)
    print('----------------------------------------------')

    best_psnr = 0.
    best_epoch = 0
    psnr_score = 0.

    model.load_state_dict(torch.load("./checkpoints/best_Epoch_LOL.pth")) #LOL best epoch
    # model.load_state_dict(torch.load("./checkpoints/best_Epoch_LCDP.pth")) #LCDP best epoch
    # model.load_state_dict(torch.load("./checkpoints/me/best_EpochS.pth")) #MSEC best epoch

    model.eval()
    psnr_score, ssim_score, niqe_score, mse_score = eval_test(model)

    print("AVG PSNR:{} AVG SSIM:{} AVG NIQE:{} AVG MSE:{}" .format(psnr_score, ssim_score, niqe_score,mse_score))
