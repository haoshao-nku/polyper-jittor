import os,sys
sys.path.append(os.getcwd())
from python.jseg.utils.registry import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PolypDataset(CustomDataset):
    """Polyp dataset.
    """
    CLASSES = ('background', 'foreground')

    PALETTE = [[255, 255, 255], [0, 0, 0]]

    def __init__(self, **kwargs):
        super(PolypDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
