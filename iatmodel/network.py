import torch
from torch import nn
import os
from .tpem import Global_pred

class Net(nn.Module):
    def __init__(self, in_dim=4, with_global=True):
        super(Net, self).__init__()
        self.with_global = with_global
        if self.with_global:
            self.global_net = Global_pred(in_channels=in_dim)

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 1) #二维 HW*C
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])  # 指定维度做内积
        image = image.view(shape)#变回原来维度
        return torch.clamp(image, 1e-8, 1.0)#张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量。

    def forward(self, low_img):

        b = low_img.shape[0]
        g1, g2,I_Mask,c = self.global_net(low_img) #I_Mask是估计光照图
        # c应用于不同通道加权
        img_high_g = torch.split(low_img[:, :, :, :], 1, dim=1)
        img_high_R = torch.stack([self.apply_color(img_high_g[0][i, :, :,:], c[0][i, :, :]) for i in range(b)], dim=0)
        img_high_G = torch.stack([self.apply_color(img_high_g[1][j, :, :,:], c[1][j, :, :]) for j in range(b)], dim=0)
        img_high_B = torch.stack([self.apply_color(img_high_g[2][k, :, :,:], c[2][k, :, :]) for k in range(b)], dim=0)
        output1 = torch.cat((img_high_R,img_high_G,img_high_B),dim=1)
        # 对于图像的不同区域应用不同伽马校正;根据张量维度i依次乘以gamma
        output = torch.where(I_Mask == 0, output1 ** g1[:, :, None, None], output1 ** g2[:, :, None, None])

        return  output

if __name__ == "__main__":
    # -----测试网络结构
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    img = torch.Tensor(1, 3, 400, 600)
    img_gray = torch.randn(1,1,400,600)
    img_test =torch.cat((img, img_gray), dim=1)

    net = Net().cuda()
    img = net(img.cuda(),img_gray.cuda())
    print(net)

    #计算参数量
    print('total parameters:', sum(param.numel() for param in net.parameters()))
    high = net(img_test)



