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
    def __init__(self, cfg, in_channels):
        super(SupportFPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM # default to 1024
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN 
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        # print(proposals)
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

class SupportFPN2ROIFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self, cfg, in_channels):
        super(SupportFPN2ROIFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM # default to 1024
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN 
        self.pooler = pooler

    def forward(self, x, proposals):
        # print(proposals)
        x = self.pooler(x, proposals)

        return x


def build_supproi_pooling(cfg, in_channels):
    # if cfg.FEW_SHOT.POOLING == 'MLP':
    #     return SupportFPN2MLPFeatureExtractor(cfg, in_channels)
    # elif cfg.FEW_SHOT.POOLING == 'ROI':
    return SupportFPN2ROIFeatureExtractor(cfg, in_channels)