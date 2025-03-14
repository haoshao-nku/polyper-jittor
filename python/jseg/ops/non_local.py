from abc import ABCMeta

import jittor as jt
from jittor import nn
import os,sys
sys.path.append(os.getcwd())
from python.jseg.utils.weight_init import constant_init, normal_init
from python.jseg.bricks import ConvModule


class _NonLocalNd(nn.Module, metaclass=ABCMeta):

    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian',
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.mode = mode

        if mode not in [
                'gaussian', 'embedded_gaussian', 'dot_product', 'concatenation'
        ]:
            raise ValueError("Mode should be in 'gaussian', 'concatenation', "
                             f"'embedded_gaussian' or 'dot_product', but got "
                             f'{mode} instead.')

        # g, theta, phi are defaulted as `nn.ConvNd`.
        # Here we use ConvModule for potential usage.
        self.g = ConvModule(self.in_channels,
                            self.inter_channels,
                            kernel_size=1,
                            conv_cfg=conv_cfg,
                            act_cfg=None)
        self.conv_out = ConvModule(self.inter_channels,
                                   self.in_channels,
                                   kernel_size=1,
                                   conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg,
                                   act_cfg=None)

        if self.mode != 'gaussian':
            self.theta = ConvModule(self.in_channels,
                                    self.inter_channels,
                                    kernel_size=1,
                                    conv_cfg=conv_cfg,
                                    act_cfg=None)
            self.phi = ConvModule(self.in_channels,
                                  self.inter_channels,
                                  kernel_size=1,
                                  conv_cfg=conv_cfg,
                                  act_cfg=None)

        if self.mode == 'concatenation':
            self.concat_project = ConvModule(self.inter_channels * 2,
                                             1,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             bias=False,
                                             act_cfg=dict(type='ReLU'))

        self.init_weights(**kwargs)

    def init_weights(self, std=0.01, zeros_init=True):
        if self.mode != 'gaussian':
            for m in [self.g, self.theta, self.phi]:
                normal_init(m.conv, std=std)
        else:
            normal_init(self.g.conv, std=std)
        if zeros_init:
            if self.conv_out.norm_cfg is None:
                constant_init(self.conv_out.conv, 0)
            else:
                constant_init(self.conv_out.norm, 0)
        else:
            if self.conv_out.norm_cfg is None:
                normal_init(self.conv_out.conv, std=std)
            else:
                normal_init(self.conv_out.norm, std=std)

    def gaussian(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = jt.matmul(theta_x, phi_x)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def embedded_gaussian(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = jt.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = jt.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def concatenation(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = jt.concat([theta_x, phi_x], dim=1)
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.size()
        pairwise_weight = pairwise_weight.view(n, h, w)
        pairwise_weight /= pairwise_weight.shape[-1]

        return pairwise_weight

    def execute(self, x):
        # Assume `reduction = 1`, then `inter_channels = C`
        # or `inter_channels = C` when `mode="gaussian"`

        # NonLocal1d x: [N, C, H]
        # NonLocal2d x: [N, C, H, W]
        # NonLocal3d x: [N, C, T, H, W]
        n = x.size(0)

        # NonLocal1d g_x: [N, H, C]
        # NonLocal2d g_x: [N, HxW, C]
        # NonLocal3d g_x: [N, TxHxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # NonLocal1d theta_x: [N, H, C], phi_x: [N, C, H]
        # NonLocal2d theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        # NonLocal3d theta_x: [N, TxHxW, C], phi_x: [N, C, TxHxW]
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)

        pairwise_func = getattr(self, self.mode)
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # NonLocal1d y: [N, H, C]
        # NonLocal2d y: [N, HxW, C]
        # NonLocal3d y: [N, TxHxW, C]
        y = jt.matmul(pairwise_weight, g_x)
        # NonLocal1d y: [N, C, H]
        # NonLocal2d y: [N, C, H, W]
        # NonLocal3d y: [N, C, T, H, W]
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, *x.size()[2:])

        output = x + self.conv_out(y)

        return output


class NonLocal1d(_NonLocalNd):

    def __init__(self,
                 in_channels,
                 sub_sample=False,
                 conv_cfg=dict(type='Conv1d'),
                 **kwargs):
        super().__init__(in_channels, conv_cfg=conv_cfg, **kwargs)

        self.sub_sample = sub_sample

        if sub_sample:
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


class NonLocal2d(_NonLocalNd):
    _abbr_ = 'nonlocal_block'

    def __init__(self,
                 in_channels,
                 sub_sample=False,
                 conv_cfg=dict(type='Conv2d'),
                 **kwargs):
        super().__init__(in_channels, conv_cfg=conv_cfg, **kwargs)

        self.sub_sample = sub_sample

        if sub_sample:
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


class NonLocal3d(_NonLocalNd):

    def __init__(self,
                 in_channels,
                 sub_sample=False,
                 conv_cfg=dict(type='Conv3d'),
                 **kwargs):
        super().__init__(in_channels, conv_cfg=conv_cfg, **kwargs)
        self.sub_sample = sub_sample

        if sub_sample:
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer
