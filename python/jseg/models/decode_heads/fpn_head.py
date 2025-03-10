import numpy as np
from jittor import nn
import os,sys
sys.path.append(os.getcwd())
from python.jseg.bricks import ConvModule

from python.jseg.ops import Upsample, resize
from python.jseg.utils.registry import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FPNHead(BaseDecodeHead):

    def __init__(self, feature_strides, **kwargs):
        super(FPNHead, self).__init__(input_transform='multiple_select',
                                      **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(scale_factor=2,
                                 mode='bilinear',
                                 align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

    def execute(self, inputs):

        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(self.scale_heads[i](x[i]),
                                     size=output.shape[2:],
                                     mode='bilinear',
                                     align_corners=self.align_corners)

        output = self.cls_seg(output)
        return output
