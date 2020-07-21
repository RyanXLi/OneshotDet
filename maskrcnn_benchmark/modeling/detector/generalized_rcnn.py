# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..supproi_pooling import build_supproi_pooling
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.layers import ROIAlign
from ..utils import cat


class SuppAlignLayer(nn.Module):
    def __init__(self, scales, output_size, sampling_ratio):
        super(SuppAlignLayer, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)

    def convert_to_roi_format(self, boxes):
        box_nums_per_img = [len(b.bbox) for b in boxes]
        assert len(set(box_nums_per_img)) == 1, box_nums_per_img
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1) # (img_id, x,y,x,y)
        return rois

    def forward(self, x, boxes):
        rois = self.convert_to_roi_format(boxes)
        result = []
        for per_level_feature, pooler in zip(x, self.poolers):
            result.append(pooler(per_level_feature, rois))
        return result


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg=cfg

        self.backbone = build_backbone(cfg)
        if cfg.FEW_SHOT.SIAMESE_BACKBONE:
            self.supp_backbone = build_backbone(cfg)
        self.use_neg_supp = cfg.FEW_SHOT.NEG_SUPPORT.TURN_ON
        self.supp_aug = cfg.FEW_SHOT.SUPP_AUG
        self.supp_aug_num = cfg.FEW_SHOT.NUM_SUPP_AUG+1 
        if self.supp_aug and cfg.FEW_SHOT.SUPP_AUG_METHOD == 'conv':
            in_dim = self.backbone.out_channels*self.supp_aug_num
            out_dim = self.backbone.out_channels
            self.supp_aug_conv = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=False),  # originally 10_15 version only has 1 conv2d layer
                )
        
        if self.use_neg_supp:
            self.rpn = build_rpn(cfg, self.backbone.out_channels)
        else:
            self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        if cfg.FEW_SHOT.SUPP_ROIALIGN:
            self.supp_pooling = SuppAlignLayer(
                                    output_size=(1,1), 
                                    scales=cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES,
                                    sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
                                    )
        else:
            self.supp_pooling = nn.AdaptiveAvgPool2d((1,1))
        if not self.cfg.MODEL.RPN_ONLY:
            self.supproi_pooling = build_supproi_pooling(cfg, self.backbone.out_channels)

        

    def batch_pooling(self, x, batch_size):
        D, C, H, W = x.shape
        x = x.view(batch_size, int(D / batch_size), C, H, W)
        x = torch.mean(x, dim=1, keepdim=False)
        return x

    def get_gt_proposals(self, targets, device):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("scores", torch.ones(len(gt_box), device=device))

        return gt_boxes

    def add_artificial_proposals(self, proposals, targets, iou_lower_bound=0.5999, required_num=3, granularity=0.1):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
            iou_lower_bound: float e.g. 0.7
            required_num: int, for each iou interval
        Note: added to the front to avoid sorting again
        """
        def box_area(box):
            return (box[2]-box[0])*(box[3]-box[1])

        def box_iou(box1, box2):
            l = torch.max(box1[0], box2[0])
            t = torch.max(box1[1], box2[1])
            r = torch.min(box1[2], box2[2])
            b = torch.min(box1[3], box2[3])
            if r - l < 0 or b - t < 0:
                return 0
            area = (r-l)*(b-t)
            iou = area / (box_area(box1) + box_area(box2) - area) 
            return iou.item()

        def isBinsFull(bins):
            for bin in bins:
                if len(bin) < required_num:
                    return False
            return True

        def random_shifts(box_list, device):

            offset = lambda thres: random.uniform(thres - 1, 1 - thres)
            box_list = box_list.convert("xyxy")
            result = []
            for box in box_list.bbox:
                bins = [[] for _ in range(int((1 - iou_lower_bound) / granularity))] 
                while not isBinsFull(bins):
                    # two points tl and br
                    x1, y1, x2, y2 = box

                    new_x1 = x1 + (x2-x1)*offset(iou_lower_bound + 0.25)
                    new_y1 = y1 + (y2-y1)*offset(iou_lower_bound + 0.25)
                    new_x2 = x2 + (x2-x1)*offset(iou_lower_bound + 0.25)
                    new_y2 = y2 + (y2-y1)*offset(iou_lower_bound + 0.25)
                    if new_x1 <= 0 or new_y1 <= 0 or new_x2 >= box_list.size[0] or new_y2 >= box_list.size[1]:
                        continue
                    new_box = (new_x1, new_y1, new_x2, new_y2)

                    iou = box_iou(box, new_box)
                    if iou < iou_lower_bound:
                        continue
                    else:
                        bin_idx = int((iou - iou_lower_bound) / granularity)
                        if len(bins[bin_idx]) < required_num:
                            bins[bin_idx].append(new_box)
                result = result + [box for item in bins for box in item] # flatten 1 layer
            result = BoxList(result, box_list.size, "xyxy").to(device)

            return cat_boxlist((result, box_list))

        assert(iou_lower_bound > 0.5)

        # Get the device we're operating on
        device = proposals[0].bbox.device

        gt_boxes = [target.copy_with_fields([]) for target in targets]


        artificial_boxes = [random_shifts(gt_box, device) for gt_box in gt_boxes]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in artificial_boxes:
            gt_box.add_field("scores", torch.ones(len(gt_box), device=device))

        proposals = [
            cat_boxlist((gt_box, proposal))
            for proposal, gt_box in zip(proposals, artificial_boxes)
        ]

        proposals = [proposal[:1000] for proposal in proposals] 

        return proposals

    def magic_combine(self, x, dim_begin, dim_end):
        combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
        return x.view(combined_shape)

    def forward(self, images, images_supp, targets, images_neg_query=None, images_neg_supp=None, device=None, target_ids=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        ####### get supp boxes ########
        images = to_image_list(images)
        images_supp = to_image_list(images_supp)
        if self.use_neg_supp:
            images_neg_supp = to_image_list(images_neg_supp)

        # TODO: after reorganize, are they still of the same size? lets check, answer is NO!!!
        # image size in image list means the original image size, while the tensors are padded for batch training

        if self.cfg.FEW_SHOT.SUPP_AUG:
            supp_image_sizes = images_supp.image_sizes
            supp_image_sizes_set = []
            for i in range(len(supp_image_sizes)):
                if i%self.supp_aug_num == 0:
                    for j in range(self.supp_aug_num-1):
                        assert supp_image_sizes[i+j] == supp_image_sizes[i+j+1]
                    supp_image_sizes_set.append(supp_image_sizes[i])
            if self.use_neg_supp:
                neg_supp_image_sizes = images_neg_supp.image_sizes
                neg_supp_image_sizes_set = []
                for i in range(len(neg_supp_image_sizes)):
                    if i%self.supp_aug_num == 0:
                        for j in range(self.supp_aug_num-1):
                            assert neg_supp_image_sizes[i+j] == neg_supp_image_sizes[i+j+1]
                        neg_supp_image_sizes_set.append(neg_supp_image_sizes[i])

        else:
            supp_image_sizes_set = images_supp.image_sizes
            if self.use_neg_supp:
                neg_supp_image_sizes_set = images_neg_supp.image_sizes
        # supp_sizes_boxlist = [BoxList([[0, 0, supp_size[0], supp_size[1]]], image_size=supp_size, mode='xyxy').to(device) for supp_size in images_supp.image_sizes]
        supp_sizes_boxlist = [BoxList([[0, 0, supp_size[0], supp_size[1]]], image_size=supp_size, mode='xyxy').to(device) for supp_size in supp_image_sizes_set]
        if self.use_neg_supp:
            neg_supp_sizes_boxlist = [BoxList([[0, 0, neg_supp_size[0], neg_supp_size[1]]], image_size=neg_supp_size, mode='xyxy').to(device) for neg_supp_size in neg_supp_image_sizes_set]

        batch_size = images.batch_size

        ####### get tensors  ########
        query_images_tensor = images.tensors
        supp_images_tensor = images_supp.tensors
        if self.use_neg_supp:
            neg_supp_images_tensor = images_neg_supp.tensors

        ####### get features and roi_pooled features ########
        features = self.backbone(query_images_tensor)
        if self.cfg.FEW_SHOT.SIAMESE_BACKBONE:
            features_supp = self.supp_backbone(supp_images_tensor)
            if self.use_neg_supp:
                features_neg_supp = self.supp_backbone(neg_supp_images_tensor)
        else:
            features_supp = self.backbone(supp_images_tensor)
            if self.use_neg_supp:
                features_neg_supp = self.backbone(neg_supp_images_tensor)

        if self.supp_aug:
            features_supp = [torch.stack(feat.split(self.supp_aug_num, dim=0), dim=0) for feat in features_supp]
            if self.use_neg_supp:
                features_neg_supp = [torch.stack(feat.split(self.supp_aug_num, dim=0), dim=0) for feat in features_neg_supp]

            if self.cfg.FEW_SHOT.SUPP_AUG_METHOD == 'avg':
                features_supp = [feat.mean(dim=1, keepdim=False) for feat in features_supp]
            elif self.cfg.FEW_SHOT.SUPP_AUG_METHOD == 'max':
                features_supp = [feat.max(dim=1, keepdim=False)[0] for feat in features_supp]
            elif self.cfg.FEW_SHOT.SUPP_AUG_METHOD == 'conv':
                features_supp = [self.magic_combine(feat,1,3) for feat in features_supp]
                features_supp = [self.supp_aug_conv(feat) for feat in features_supp]
                if self.use_neg_supp:
                    features_neg_supp = [self.magic_combine(feat,1,3) for feat in features_neg_supp]
                    features_neg_supp = [self.supp_aug_conv(feat) for feat in features_neg_supp]

        if not self.cfg.MODEL.RPN_ONLY:
            features_supp_roipooled = self.supproi_pooling(features_supp, supp_sizes_boxlist)
            if self.use_neg_supp:
                features_neg_supp_roipooled = self.supproi_pooling(features_neg_supp, neg_supp_sizes_boxlist)

        # support feature pooling stupidly
        if not self.cfg.FEW_SHOT.SUPP_ROIALIGN:
            features_supp_pooled = [self.supp_pooling(feature_supp) for feature_supp in features_supp]
        else:
            features_supp_pooled = self.supp_pooling(features_supp, supp_sizes_boxlist)
        features_supp_pooled = [self.batch_pooling(feature_supp_pooled, batch_size) for feature_supp_pooled in features_supp_pooled]
        combined_features = []
        for i in range(len(features)):
            B, C, Dim1, Dim2 = features[i].shape
            expanded_supp = features_supp_pooled[i].expand(-1, -1, Dim1, Dim2) 
            combined_features.append(features[i] * expanded_supp)
        proposals, proposal_losses = self.rpn(images, combined_features, targets)

        if self.roi_heads:
            if self.use_neg_supp:
                x, result, detector_losses = self.roi_heads(features, proposals, targets, features_supp_roipooled, features_neg_supp_roipooled)
            else:
                x, result, detector_losses = self.roi_heads(features, proposals, targets, features_supp_roipooled, target_ids=target_ids)

        else:
            # RPN-only models don't have roi_heads
            x = combined_features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
