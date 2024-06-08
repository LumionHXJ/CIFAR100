from typing import Dict, Optional
from mmpretrain.registry import TRANSFORMS
from timm.data.auto_augment import auto_augment_policy, AutoAugment
from mmcv.transforms import BaseTransform
from PIL import Image
import numpy as np

@TRANSFORMS.register_module()
class TIMM_AutoAugment(BaseTransform):
    def __init__(self, name='v0') -> None:
        super().__init__()
        self.policy = auto_augment_policy(name)
        self.trans = AutoAugment(self.policy)

    def transform(self, results: Dict) -> Optional[Dict]:
        """Randomly choose a transform to apply."""
        pil_img = Image.fromarray(results['img'])
        results['img'] = np.array(self.trans(pil_img))
        return results

    def __repr__(self) -> str:
        policies_str = ''
        for sub in self.policies:
            policies_str += '\n    ' + ', \t'.join([t['type'] for t in sub])

        repr_str = self.__class__.__name__
        repr_str += f'(policies:{policies_str}\n)'
        return repr_str  