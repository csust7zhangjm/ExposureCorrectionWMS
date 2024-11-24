import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
import os
from .blocks import Mlp
from lib.utils import e_gray,aMask

class query_Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Parameter(torch.ones((1, 10, dim)), requires_grad=True)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)#全连接层
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)#防止过拟合
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)#softmax输出概率
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 10, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class query_SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim) #group控制分组卷积，每层分组卷积
        self.norm1 = norm_layer(dim)
        self.attn = query_Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # DropPath/drop_path 是一种正则化手段，其效果是将深度学习模型中的多分支结构随机”删除“
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        #使用两个MLP分离最后的分离
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)  #(b,c,h*w)--(b,hw,c)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_embedding, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class Global_pred(nn.Module):
    def __init__(self, in_channels=4, out_channels=64, num_heads=4):
        super(Global_pred, self).__init__()
        self.gamma_base1 = nn.Parameter(torch.ones((1)), requires_grad=True)
        self.gamma_base2 = nn.Parameter(torch.ones((1)), requires_grad=True)
        self.base_1 = nn.Parameter(torch.ones(8), requires_grad=True)
        self.base_2 = nn.Parameter(torch.ones(8), requires_grad=True)
        self.base_3 = nn.Parameter(torch.ones(8), requires_grad=True)

        # main blocks
        self.conv_large = conv_embedding(in_channels, out_channels)
        self.generator = query_SABlock(dim=out_channels, num_heads=num_heads)
        #全连接层，参数为输入和输出
        self.gamma_linear = nn.Linear(out_channels, 1)
        self.color_linear = nn.Linear(out_channels, 3)

        self.apply(self._init_weights)
        # self.Att_img = VisualAttentionNetwork()

        for name, p in self.named_parameters():
            if name == 'generator.attn.v.weight':
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        I = e_gray(x)
        # 自适应Mask生成
        I_Mask = aMask(I)

        x = torch.cat((x,I),dim=1) #将光照图加入训练
        x = self.conv_large(x)
        x = self.generator(x)
        #==========基于光照图学习两个gamma值用于图像不同区域
        gamma1, gamma2, color = x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1), x[:, 2:]
        gamma1 = self.gamma_linear(gamma1).squeeze(-1) + self.gamma_base1
        gamma2 = self.gamma_linear(gamma2).squeeze(-1) + self.gamma_base2
        color = self.color_linear(color).transpose(2, 1)  # [B,1,8]
        colors = list(torch.split(color, 1, dim=1))
        colors[0] = colors[0] + self.base_1
        colors[1] = colors[1] + self.base_2
        colors[2] = colors[2] + self.base_3

        return gamma1,gamma2,I_Mask,colors

import cv2
import numpy
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    img = cv2.imread("D:/study/code/datasets/LOL/our485/Low/2.png")
    low_image = numpy.expand_dims(img.transpose((2, 0, 1)), 0)
    low_image = torch.Tensor(low_image)
    l_image= torch.randn(1, 3, 600, 400)
    i_image= torch.randn(1, 1, 600, 400)

    global_net = Global_pred().cuda()
    gamma, color = global_net(l_image.cuda(),i_image.cuda())
    print(gamma.shape, color.shape)