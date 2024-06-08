import torch.nn as nn
from mmpretrain.registry import MODELS
from timm.loss import SoftTargetCrossEntropy
import torch
from mmpretrain.models.losses import convert_to_one_hot

@MODELS.register_module()
class TIMM_LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, 
                 label_smooth=0.1,
                 num_classes=100):
        super().__init__()
        self.ce = SoftTargetCrossEntropy()
        self.num_classes = num_classes
        self._eps = label_smooth

    def generate_one_hot_like_label(self, label):
        """This function takes one-hot or index label vectors and computes one-
        hot like label vectors (float)"""
        # check if targets are inputted as class integers
        if label.dim() == 1 or (label.dim() == 2 and label.shape[1] == 1):
            label = convert_to_one_hot(label.view(-1, 1), self.num_classes)
        return label.float()
    
    def original_smooth_label(self, one_hot_like_label):
        assert self.num_classes > 0
        smooth_label = one_hot_like_label * (1 - self._eps)
        smooth_label += self._eps / self.num_classes
        return smooth_label
    
    def forward(self,
                cls_score: torch.Tensor,
                label: torch.Tensor):
        label = self.generate_one_hot_like_label(label)
        smooth_label = self.original_smooth_label(label)
        return self.ce(cls_score, smooth_label)