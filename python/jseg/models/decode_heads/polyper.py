


import jittor as jt
from jittor import nn
import os,sys
sys.path.append(os.getcwd())
from python.jseg.bricks import ConvModule

import numbers

import cv2 as cv
import numpy as np
from python.jseg.utils.registry import HEADS
from .decode_head import BaseDecodeHead

from python.jseg.ops import resize
def erosion_to_dilate(output):
    z = output.numpy()  # 转换为 NumPy 数组
    z = np.where(z > 0.3, 1.0, 0.0)  # 二值化分割结果
    z = jt.array(z)  # 转回 Jittor 张量

    kernel = np.ones((4, 4), np.uint8)  # 定义卷积核
    maskd = np.zeros_like(output.numpy())  # 结果数组
    maske = np.zeros_like(output.numpy())  # 结果数组

    for i in range(output.shape[0]):
        y = z[i].permute(1, 2, 0)
        erosion = y.numpy()
        dilate = y.numpy()

        # 转换为uint8格式以使用OpenCV操作
        dilate = np.array(dilate, dtype='uint8')
        erosion = np.array(erosion, dtype='uint8')

        # 使用OpenCV进行腐蚀和膨胀操作
        erosion = cv.erode(erosion, kernel, iterations=4)
        dilate = cv.dilate(dilate, kernel, iterations=4)

        # 转换回 Jittor 张量并调整维度
        mask1 = jt.array(dilate - erosion).unsqueeze(-1).permute(2, 0, 1)
        mask2 = jt.array(erosion).unsqueeze(-1).permute(2, 0, 1)

        # 保存结果
        maskd[i] = mask1.numpy()
        maske[i] = mask2.numpy()

    # 转回 Jittor 张量并移动到 GPU
    maskd = jt.array(maskd)
    maske = jt.array(maske)

    return maskd, maske




def to_3d(x):
    b, c, h, w = x.shape
    return x.reshape(b, c, h * w).transpose(0, 2, 1)  # b (h w) c

def to_4d(x, h, w):
    b, hw, c = x.shape
    return x.transpose(0, 2, 1).reshape(b, c, h, w)  # b c h w

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.weight = jt.zeros(normalized_shape)  # Jittor参数初始化

    def execute(self, x):  # Jittor使用execute替代forward
        sigma = x.var(-1, keepdims=True, unbiased=False)
        return x / jt.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = jt.ones(normalized_shape)
        self.bias = jt.zeros(normalized_shape)
        self.normalized_shape = normalized_shape

    def execute(self, x):
        mu = x.mean(-1, keepdims=True)
        sigma = x.var(-1, keepdims=True, unbiased=False)
        return (x - mu) / jt.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def execute(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class RefineAttention(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type):
        super().__init__()
        self.num_heads = num_heads
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def execute(self, x, mask_d, mask_e):
        b, c, h, w = x.shape
        x = self.norm(x)
        y1 = x * (1 - mask_e)
        y2 = x * (1 - mask_d)
        out_sa = x.clone()
        
        with jt.no_grad():
            for i in range(b):
                z_d = []
                z_e = []
                pos_d = jt.nonzero((mask_d[i][0] == 1).bool()).long()
                pos_e = jt.nonzero((mask_e[i][0] == 1).bool()).long()

                if len(pos_d) == 0 or len(pos_e) == 0:
                    continue  # 跳过空索引的情况

                for j in range(c):
                    h_idx = pos_d[:, 0]
                    w_idx = pos_d[:, 1]
                    z_d.append(x[i, j, h_idx, w_idx])

                    h_idx_e = pos_e[:, 0]
                    w_idx_e = pos_e[:, 1]
                    z_e.append(x[i, j, h_idx_e, w_idx_e])

                z_d = jt.stack(z_d)
                z_e = jt.stack(z_e)

                z_e = z_e.reshape(self.num_heads, -1, c // self.num_heads)
                k1 = z_e
                v1 = z_e

                z_d = z_d.reshape(self.num_heads, -1, c // self.num_heads)
                q1 = z_d

                norm1 = nn.LayerNorm(q1.shape[-1])
                q1 = norm1(q1)
                k1 = norm1(k1)

                attn1 = (q1 @ k1.transpose(-2, -1))
                attn1 = attn1.softmax(dim=-1)
                out1 = (attn1 @ v1) + q1
                out1 = out1.transpose(0, 2, 1).reshape(-1, out1.shape[1])

                for j in range(c):
                    out_sa[i, j, pos_d[:, 0], pos_d[:, 1]] = out1[j]

        y2 = y2.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k2 = y2
        v2 = y2

        y1 = y1.reshape(b, self.num_heads, c // self.num_heads, h * w)
        q2 = y1

        norm2 = nn.LayerNorm(q2.shape[-1])
        q2 = norm2(q2)
        k2 = norm2(k2)
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out2 = (attn2 @ v2) + q2
        out2 = out2.reshape(b, self.num_heads * (c // self.num_heads), h, w)

        out = x + out_sa + out2
        out = self.project_out(out)
        return out




@HEADS.register_module()
class PolyperHead(BaseDecodeHead):
    def __init__(self,in_channels,image_size,heads,
                 **kwargs):
        super(PolyperHead, self).__init__(in_channels,input_transform = 'multiple_select',**kwargs)
        self.image_size = (image_size,image_size)

        self.refine1 = RefineAttention(in_channels[0],heads,LayerNorm_type = 'WithBias')
        self.refine2 = RefineAttention(in_channels[1],heads,LayerNorm_type = 'WithBias')
        self.refine3 = RefineAttention(in_channels[2],heads,LayerNorm_type = 'WithBias')
        self.refine4 = RefineAttention(in_channels[3],heads,LayerNorm_type = 'WithBias')
        self.align1 = ConvModule(
            in_channels[3],
            in_channels[2],
            1)
        self.align2 = ConvModule(
            in_channels[2],
            in_channels[1],
            1)        
        self.align3 = ConvModule(
            in_channels[1],
            in_channels[0],
            1 )

    def execute(self, inputs):
        inputs = [resize(
                level,
                size=self.image_size,
                mode='bilinear'
            ) for level in inputs]        

        #stage4
        y3 = inputs[3]
        #stage3
        conv_y1 = self.align1(y3)
        y2 = inputs[2]+conv_y1
        #stage3
        conv_y2 = self.align2(y2)
        y1 = inputs[1]+conv_y2           
        #stage3
        conv_y3 = self.align3(y1)
        y0 = inputs[0]+conv_y3    #y0   [2,96,128,128]

        mask_d,mask_e = erosion_to_dilate(self.cls_seg(y0))
        #stage4
        y3 = self.refine4(y3, mask_d,mask_e)
        #stage3
        conv_y1 = self.align1(y3)
        y2 = self.refine3(y2+conv_y1,mask_d,mask_e)
        #stage3
        conv_y2 = self.align2(y2)
        y1 = self.refine2(y1+conv_y2,mask_d,mask_e)            
        #stage3
        conv_y3 = self.align3(y1)
        y0 = self.refine1(y0+conv_y3,mask_d,mask_e)

        output = self.cls_seg(y0)
        return output