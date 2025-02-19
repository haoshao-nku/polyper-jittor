from jittor.nn import Dropout, DropPath
import os,sys
sys.path.append(os.getcwd())
from python.jseg.utils.registry import DROPOUT_LAYERS, build_from_cfg

DROPOUT_LAYERS.register_module(name='Dropout', module=Dropout)
DROPOUT_LAYERS.register_module(name='DropPath', module=DropPath)


def build_dropout(cfg, **default_args):
    """Builder for drop out layers."""
    return build_from_cfg(cfg, DROPOUT_LAYERS, **default_args)
