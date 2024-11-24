import matplotlib.pyplot as plt
import data_load
import os
import torch.optim
import argparse
import torch.nn as nn
import torch.nn.functional as F
from iatmodel.dbfm import DB_iat
from iatmodel.utils import validation
from lib.utils import Colorloss,PerceptualLoss,MS_SSIMLoss

# Setting
# LOL:
#   lr:0.0001,batch size:2;lr_scheduler:MultiStepLR;epoch:400

# lcdp:
#   lr:0.0001,batch size:2;lr_scheduler:MultiStepLR;epoch:200


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default=0)
    parser.add_argument("--normalize", action="store_false", help="Default Normalize in LOL training.")
    parser.add_argument('--train_LL_folder', type=str, default="D:/study/code/datasets/LOL/our485/Low/")
    # parser.add_argument('--train_LL_folder', type=str, default="D:/study/code/datasets/LOL/eval15/High/")
    # parser.add_argument('--train_LL_folder', type=str, default="D:/study/code/datasets/LOL_v2/Train/Low")
    # parser.add_argument('--train_LL_folder', type=str, default="D:/study/code/datasets/lcdp_dataset/input")
    # parser.add_argument('--train_LL_folder', type=str,default="D:/study/code/datasets/fivek/input")
    parser.add_argument('--img_val_path', type=str, default="D:/study/code/datasets/LOL/eval15/Low/")
    # parser.add_argument('--img_val_path', type=str, default="D:/study/code/datasets/lcdp_dataset/test-input")
    # parser.add_argument('--img_val_path', type=str, default="D:/study/code/datasets/fivek/test")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--milestones', type=str, default='250,280')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.05)
    parser.add_argument('--pretrain_dir', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--display_iter', type=int, default=1)
    parser.add_argument('--snapshots_folder', type=str, default="./checkpoints/")
    parser.add_argument('--resize', default=False)
    config = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)

    print(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    ssim_high = 0
    psnr_high = 0
    if not os.path.exists(config.snapshots_folder):
        os.makedirs(config.snapshots_folder)

    #========== Model Setting
    model = DB_iat().cuda()

    # ========== Data Setting
    img, val_img = data_load.load_init(config)
    # ========== Model loading
    if config.pretrain_dir is not None:
        model.load_state_dict(torch.load(config.pretrain_dir))#iat
        ssim_high, psnr_high = validation(model, val_img)

    # Loss & Optimizer Setting & Metric
    # =============================#
    #         Loss function        #
    #    VGG16/l1/l1_SMOOTH/Tvloss
    # =============================#

    L1_loss = nn.L1Loss()
    L1_smooth_loss = F.smooth_l1_loss
    color_loss = Colorloss()
    p_loss = PerceptualLoss()

    #设置学习率策略
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config.milestones.split(',')], gamma=config.gamma)

    device = next(model.parameters()).device
    print('the device is:', device)

    model.train()
    print('######## Start IAT Training #########')
    for epoch in range(config.num_epochs):
        print('=============the epoch is:', epoch)

        for iteration, imgs in enumerate(img):
            low_img, high_img= imgs[0].cuda(), imgs[1].cuda()

            #清除梯度
            optimizer.zero_grad()
            model.train()

            enhance_img = model(low_img)

            #=======================================#
            #==============   LOSS   ===============#
            #=======================================#

            loss = 0.6 * L1_loss(enhance_img, high_img) + 0.5 * L1_smooth_loss(enhance_img,high_img) + 0.2 * MS_SSIMLoss(enhance_img, high_img) + 0.1 * p_loss(enhance_img, high_img) + 0.8 * color_loss(high_img, enhance_img)

            # ======= Loss print
            print("Epoch:{},iteration:{} Loss:{}".format(epoch, iteration, loss))

            # 梯度更新
            loss.backward()
            optimizer.step()

        model.eval()
        SSIM_mean,PSNR_mean = validation(model, val_img)
        # log
        with open(config.snapshots_folder+'/log.txt', 'a+') as f:
            f.write('epoch' + str(epoch) + ':' + 'the SSIM is' + str(SSIM_mean) + 'the PSNR is' + str(PSNR_mean) + 'LOSS:' + str(
                format(loss)) + '\n')

        if SSIM_mean > ssim_high:
            ssim_high = SSIM_mean
            print('the highest SSIM value is:', str(ssim_high))
            torch.save(model.state_dict(), os.path.join(config.snapshots_folder, "best_Epoch_SSIM"+ '.pth'))
        if PSNR_mean > psnr_high:
            psnr_high = PSNR_mean
            print('the highest PSNR value is:', str(PSNR_mean))
            torch.save(model.state_dict(), os.path.join(config.snapshots_folder, "best_Epoch_PSNR"+ '.pth'))

        scheduler.step()
        f.close()


