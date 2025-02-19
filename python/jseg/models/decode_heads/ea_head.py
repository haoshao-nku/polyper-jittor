from .decode_head import BaseDecodeHead
import os,sys
sys.path.append(os.getcwd())
from python.jseg.utils.registry import HEADS
from python.jseg.ops import External_attention


@HEADS.register_module()
class EAHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(EAHead, self).__init__(**kwargs)
        self.ea = External_attention(self.in_channels, self.channels)

    def execute(self, inputs):
        x = self._transform_inputs(inputs)
        x = self.ea(x)
        output = self.cls_seg(x)
        return output
