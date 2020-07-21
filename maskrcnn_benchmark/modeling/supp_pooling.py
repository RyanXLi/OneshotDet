import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc

class SupportFPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self, cfg, in_channels, out_channels):
        super(SupportFPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.SUPP_POOLING.POOLER_RESOLUTION
        scales = cfg.MODEL.SUPP_POOLING.POOLER_SCALES
        sampling_ratio = cfg.MODEL.SUPP_POOLING.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.SUPP_POOLING.MLP_HEAD_DIM # default to 1024
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, out_channels, use_gn)
        self.out_channels = out_channels

    def forward(self, x, proposals):
        # print(proposals)
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

def build_supp_pooling(cfg, in_channels, out_channels):
    return SupportFPN2MLPFeatureExtractor(cfg, in_channels, out_channels)