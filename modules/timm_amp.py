import torch
from typing import Union

from mmpretrain.registry import OPTIM_WRAPPERS
from mmengine.optim.optimizer import AmpOptimWrapper
from timm.utils import NativeScaler

@OPTIM_WRAPPERS.register_module()
class TIMM_AmpOptimWrapper(AmpOptimWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_scaler = NativeScaler()._scaler