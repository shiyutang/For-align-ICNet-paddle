"""Base Model for Semantic Segmentation"""
import paddle.nn as nn
from .base_models.resnetv1b import resnet50_v1b, resnet50_v1s

__all__ = ['SegBaseModel']

class SegBaseModel(nn.Layer):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50v1s'; 'resnet50').
    """

    def __init__(self, nclass, backbone='resnet50', pretrained_base=False, **kwargs):

        backbone='resnet50'
        super(SegBaseModel, self).__init__()
        dilated = True
        self.nclass = nclass
        if backbone == 'resnet50':
            self.pretrained = resnet50_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet50v1s':
            self.pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        import pdb; pdb.set_trace()

        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        
        return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        return pred
