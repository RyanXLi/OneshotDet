# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.modeling.make_layers import make_fc


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels): # in_channels = 256
        super(ROIBoxHead, self).__init__()
        ### if  cfg.FEW_SHOT.POOLING == 'ROI' and cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR == 'FPN2ROIFeatureExtractor'
        ### output is simply pooled features, we will aggregate them together in feature aggreg layer and fc layer
        ### otherwise the output is a vector having gone through the fc layer
        self.cfg=cfg
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)

        ### fuse query and support features if necessary
        self.comparison_method = cfg.FEW_SHOT.SECOND_STAGE_METHOD
        # assert(self.comparison_method == 'matching' or self.comparison_method == 'concat')
        self.use_neg_supp = cfg.FEW_SHOT.NEG_SUPPORT.TURN_ON
        out_channel = self.feature_extractor.out_channels # 256

        num_in_layers = 1
        if (self.comparison_method == 'rn') or \
            (self.comparison_method == 'concat' and not self.use_neg_supp) or \
            (self.comparison_method == 'matching' and self.use_neg_supp):
            num_in_layers = 2
        elif self.comparison_method == 'concat' and self.use_neg_supp :
            num_in_layers = 2 # originally 3, new version keeps the same

        compressed_dim = out_channel
        # if self.comparison_method == 'concat' and self.use_neg_supp :
        #     compressed_dim = int(out_channel*1.5) #  256*1.5, usually 256
        if num_in_layers > 1 and not self.cfg.FEW_SHOT.LINEAR_FUSION:
            self.compress_dim_conv = nn.Sequential(
                        nn.Conv2d(out_channel * num_in_layers, out_channel * num_in_layers, 1),
                        nn.GroupNorm(32, out_channel * num_in_layers),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(out_channel * num_in_layers, compressed_dim, 1),
                        nn.GroupNorm(32, compressed_dim),
                        nn.LeakyReLU(0.2),
                    )
            for l in self.compress_dim_conv:
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)

        ### in roi pooling mode, we need to aggregate pooled features of query and support together 
        ### and send it to the fc layer, resulting in the final vector\
        ### in LSTD paper, the effect of using conv block instead of fc layer was claimed to be better 
        ### for security reasons, we use convblock + fc layer
        # if cfg.FEW_SHOT.POOLING == 'ROI':
        if not self.cfg.FEW_SHOT.LINEAR_FUSION:
            self.feature_aggreg = nn.Sequential(
                    nn.Conv2d(compressed_dim, int(compressed_dim/2), 3, 1, 1),  # 128
                    nn.GroupNorm(32, int(compressed_dim/2)),
                    nn.LeakyReLU(0.2),
                )
        else:
            self.feature_aggreg = nn.Sequential(
                    nn.Conv2d(2*compressed_dim, int(compressed_dim/2), 3, 1, 1),  # 128
                    nn.GroupNorm(32, int(compressed_dim/2)),
                    nn.LeakyReLU(0.2),
                )
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(int(compressed_dim/2) * resolution**2, representation_size) # 64*49, 1024, False
        self.fc7 = make_fc(representation_size, representation_size) 
        predictor_in_channels = representation_size

        self.predictor = make_roi_box_predictor(cfg, predictor_in_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)


    def forward(self, features, proposals, targets=None, features_supp_roipooled=None, features_neg_supp_roipooled=None, neg_query_features=None, target_ids=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. 
        # in MLP mode, feature_extractor generally corresponds to the pooler + heads
        # in ROI mode, it outputs simply roi aligned features
        x = self.feature_extractor(features, proposals) # (bs, num_rois_per_img, c, w, h)

        if self.cfg.FEW_SHOT.GT_PROPOSAL_ONLY:
            x  = torch.cat([x, neg_query_features], dim=1)
        
        bs, num_rois, c, w, h = x.size()

        assert features_supp_roipooled.size(1) == 1, features_supp_roipooled.size()
        features_supp_roipooled = features_supp_roipooled.view(bs, -1, c, w, h)
        # print(features_supp_roipooled.shape)

        total_class_logits = []
        total_box_regression = []

        x_copy = x  
        for idx_aug in range(features_supp_roipooled.size(1)):
            x = x_copy
            per_aug_supp_features_supp_roipooled = features_supp_roipooled[:, [idx_aug], :, :, :]
            
            expanded_supp = per_aug_supp_features_supp_roipooled.expand_as(x).contiguous().view(-1, c, w, h)
            if self.use_neg_supp:
                expanded_neg_supp = features_neg_supp_roipooled.expand_as(x).contiguous().view(-1, c, w, h)

            x = x_copy.view(-1, c, w, h)
            
            if self.comparison_method != 'rn':
                if self.comparison_method == 'concat' and self.use_neg_supp: 
                    x_neg = torch.cat((x, expanded_neg_supp), dim=1)
                    x_neg = self.compress_dim_conv(x_neg)  #change by linz

                    x = torch.cat((x, expanded_supp), dim=1)
                    x = self.compress_dim_conv(x)  #change by linz

                elif self.comparison_method == 'concat' and not self.use_neg_supp: #change by linz
                    if self.cfg.FEW_SHOT.REVERSE_ORDER: 
                        # for reverse testing
                        x_rev = torch.cat((expanded_supp, x), dim=1)
                        x_rev = self.compress_dim_conv(x_rev)

                    x = torch.cat((x, expanded_supp), dim=1)
                    if not self.cfg.FEW_SHOT.LINEAR_FUSION:
                        x = self.compress_dim_conv(x)

                x = self.feature_aggreg(x)
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc6(x))
                x = F.relu(self.fc7(x))
                class_logits, box_regression = self.predictor(x)
                if self.use_neg_supp:
                    x_neg = self.feature_aggreg(x_neg)
                    x_neg = x_neg.view(x_neg.size(0), -1)
                    x_neg = F.relu(self.fc6(x_neg))
                    x_neg = F.relu(self.fc7(x_neg))
                    neg_class_logits, neg_box_regression = self.predictor(x_neg)


                if self.cfg.FEW_SHOT.REVERSE_ORDER:
                    # for reverse testing
                    x_rev = self.feature_aggreg(x_rev)
                    x_rev = x_rev.view(x_rev.size(0), -1)
                    x_rev = F.relu(self.fc6(x_rev))
                    x_rev = F.relu(self.fc7(x_rev))
                    # final classifier that converts the features into predictions
                    rev_class_logits, rev_box_regression = self.predictor(x_rev)
                else:
                    rev_class_logits = None

                if not self.training:
                    total_class_logits.append(class_logits)
                    total_box_regression.append(box_regression)

                else:
                    if self.use_neg_supp:
                        loss_classifier, loss_box_reg, cls_suppress = self.loss_evaluator(
                            [class_logits], [box_regression], [neg_class_logits]
                        ) # linz
                        loss_classifier *= 5
                        loss_box_reg *= 2.5
                        cls_suppress *= 2.5

                        loss_dict = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_cls_suppress=cls_suppress)
                    else:
                        loss_classifier, loss_box_reg = self.loss_evaluator(
                            [class_logits], [box_regression]
                        ) # linz
                        loss_classifier *= 5
                        loss_box_reg *= 2.5
                        loss_dict = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)

                    return (
                        x,
                        proposals,
                        loss_dict
                    )
            elif self.comparison_method == 'rn': # only rn in neg support mode
                assert self.use_neg_supp
                x_pos = torch.cat((x, expanded_supp), dim=1)
                x_pos = self.compress_dim_conv(x_pos)
                x_pos = self.feature_aggreg(x_pos)
                x_pos = x_pos.view(x.size(0), -1)
                x_pos = F.relu(self.fc6(x_pos))
                x_pos = F.relu(self.fc7(x_pos))
                pos_class_logits, pos_box_regression = self.predictor(x_pos)

                x_neg = torch.cat((x, expanded_neg_supp), dim=1)
                x_neg = self.compress_dim_conv(x_neg)
                x_neg = self.feature_aggreg(x_neg)
                x_neg = x_neg.view(x.size(0), -1)
                x_neg = F.relu(self.fc6(x_neg))
                x_neg = F.relu(self.fc7(x_neg))
                neg_class_logits, neg_box_regression = self.predictor(x_neg)

                x = x_pos
                if not self.training: # currently only naively testing 
                    result = self.post_processor((pos_class_logits, pos_box_regression), proposals, (neg_class_logits, neg_box_regression))
                    return x, result, {}
                else:
                    pos_loss_classifier, pos_loss_box_reg = self.loss_evaluator(
                        [pos_class_logits], [pos_box_regression], gt_label=1
                    )
                    neg_loss_classifier, neg_loss_box_reg = self.loss_evaluator(
                        [neg_class_logits], [neg_box_regression], gt_label=2
                    )
                    loss_classifier = pos_loss_classifier# + neg_loss_classifier)/2
                    loss_box_reg = pos_loss_box_reg# + neg_loss_box_reg)/2
                    
                    return (
                        x,
                        proposals,
                        dict(loss_pos_classifier=pos_loss_classifier*5, loss_pos_box_reg=pos_loss_box_reg*10,
                            loss_neg_classifier=neg_loss_classifier*5, loss_neg_box_reg=neg_loss_box_reg*10)
                    )   

        if len(total_class_logits) > 1:
            total_class_logits = torch.stack(total_class_logits, dim=0)
            _, w1, h1 = total_class_logits.shape
            total_box_regression = torch.stack(total_box_regression, dim=0)
            _, w2, h2 = total_box_regression.shape
            cls_idx = torch.argmax(total_class_logits, dim=0)
            class_logits = total_class_logits[cls_idx, torch.arange(w1)[:, None], torch.arange(h1)]
            dim_a, dim_b = class_logits.shape
            box_idx = cls_idx[:, :, None].expand(dim_a, dim_b, 4)
            batch_size, num_cls, four = box_idx.shape 
            box_idx = box_idx.contiguous().view(batch_size, num_cls * four)#.type(torch.uint8)

            box_regression = total_box_regression[box_idx, torch.arange(w2)[:, None], torch.arange(h2)] 
        else:
            class_logits = total_class_logits[0]
            box_regression = total_box_regression[0]
        
        result = self.post_processor((class_logits, box_regression), proposals, target_ids=target_ids)
        return x, result, {}


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)

