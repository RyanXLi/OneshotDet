# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss, SigmoidFocalLoss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self, 
        cfg,
        proposal_matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.cfg = cfg
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        if cfg.FEW_SHOT.SECOND_STAGE_CLS_LOSS == 'focal_loss':
            self.cls_loss_func = SigmoidFocalLoss(
                cfg.MODEL.FCOS.LOSS_GAMMA,
                cfg.FEW_SHOT.SECOND_STAGE_LOSS_ALPHA
            )

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal) # (M, N)
        matched_idxs = self.proposal_matcher(match_quality_matrix) # -1, -2  (N)  
        ###############################
        ###   create soft labels    ### currently only assume 1 class
        ###############################
        if self.cfg.FEW_SHOT.SOFT_LABELING:
            match_iou_matrix = match_quality_matrix.t() # (N, M)
            # assert match_iou_matrix.size(1) == 2, 'only supporting 1 classes right now, but received {} classes'.format(match_iou_matrix.size(1))
            # assert torch.sum(match_iou_matrix[torch.nonzero(matched_idxs<1), 1]) == 0, \
            #             ['positive class column of non-positive prediction should all be zero ! ',
            #                 torch.sum(match_iou_matrix[torch.nonzero(matched_idxs>0), 1])]
            matched_idxs_temp = matched_idxs.clone()
            matched_idxs_temp[matched_idxs_temp<0] = 0
            match_iou = match_iou_matrix[torch.arange(len(matched_idxs_temp)).long(), matched_idxs_temp].clone()
            matched_idxs_invalid_inds = torch.nonzero(matched_idxs<0)
            match_iou[matched_idxs_invalid_inds] = 0

            # match_iou_matrix_max, _ = match_iou_matrix.max(dim=1)
            # match_iou_big_iou = torch.nonzero(match_iou_matrix_max>0.5).squeeze(1)
            # print(match_iou_big_iou)
            # print('iiou', torch.cat([match_iou[match_iou_big_iou].unsqueeze(1), match_iou_matrix[match_iou_big_iou]], dim=1))

        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        if self.cfg.FEW_SHOT.SOFT_LABELING:
            matched_targets.add_field('soft_labels', match_iou)
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def soft_labeling_function(self, t):
        '''
            discrete: t >= 0.5
            linear  : t
            transLinear: (0.2*t+0.8) * (t>=0.5) +                0.5 ~ 1    -->  0.9 ~ 1
                         (2.25*t-0.225) * (t>=0.1) * (t<0.5) +   0.1 ~ 0.5  -->    0 ~ 0.9
                         0                                       0 ~ 0.1      
            trans4thLinear: (0.2*t + 0.8) * (t>=0.5) +                0.5 ~ 1    -->  0.9 ~ 1
                            0.9*(2*t)**4                                0 ~ 0.5  -->    0 ~ 0.9

        '''        
        if self.cfg.FEW_SHOT.SOFT_LABELING_FUNC == 'discrete':
            return (t>=0.5).float()
        elif self.cfg.FEW_SHOT.SOFT_LABELING_FUNC == 'linear':
            return t
        elif self.cfg.FEW_SHOT.SOFT_LABELING_FUNC == 'transLinear':        # transitional linear
            upper  = (0.2*t+0.8) * (t>=0.5).float()
            middle = (2.25*t-0.225) * (t>=0.1).float() * (t<0.5).float()
            lower  = 0
            return upper+middle+lower
        elif self.cfg.FEW_SHOT.SOFT_LABELING_FUNC == 'trans4thLinear':        # transitional 4th order linear
            upper  = (0.2*t+0.8) * (t>=0.5).float()
            lower  = 0.9*((2*t)**4) * (t<0.5).float()
            return upper+lower

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        if self.cfg.FEW_SHOT.SOFT_LABELING:
            soft_labels = []

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals( # -1, -2
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            if self.cfg.FEW_SHOT.SOFT_LABELING:
                soft_labels_per_image = self.soft_labeling_function(matched_targets.get_field("soft_labels")) 
                soft_labels.append(soft_labels_per_image)

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        if self.cfg.FEW_SHOT.SOFT_LABELING:
            return labels, soft_labels, regression_targets
        return labels, regression_targets

    # def sample_proposal_per_box(self, box, image_size, iou_range=(0.5, 0.9), num=100):
    #     '''
    #         receive box (x1, y1, x2, y2)
    #     '''
    #     def uniSample(a, b):
    #         return torch.rand(1).squeeze()*(b-a) + a
    #     def b_area(box):
    #         return (box[2]-box[0])*(box[3]-box[1])
    #     def b_iou(box1, box2):
    #         l = torch.max(box1[0], box2[0])
    #         t = torch.max(box1[1], box2[1])
    #         r = torch.min(box1[2], box2[2])
    #         b = torch.min(box1[3], box2[3])
    #         area = (r-l)*(b-t)
    #         return  area / ( b_area(box1) + b_area(box2) - area ) 
       
    #     x1, y1, x2, y2 = box
    #     W, H = image_size 
    #     assert  x1>=0 and x1<=x2 and x2<W and \
    #             y1>=0 and y1<=y2 and y2<H, 'input box not valid {} {} {} {}'.format(x1, y1, x2, y2)
    #     center_x, center_y = (x1+x2)/2, (y1+y2)/2
    #     box_w = x2-x1+1
    #     box_h = y2-y1+1
    #     proposals = []

    #     while len(proposals) < num:
    #         rand_iou = uniSample(*iou_range) # restrict to iou range
    #         min_iou_term = 2*rand_iou/(rand_iou+1)
    #         xc_prop = uniSample(0, 0.3333)  # restrict to 0~1/3
    #         yc_prop_min = torch.max(0., min_iou_term)
    #         yc_prop_max = torch.min(0.3333, min_iou_term)
    #         if yc_prop_max <= yc_prop_min:
    #             continue
    #         yc_prop = uniSample(yc_prop_min, yc_prop_max)

    #         xc_dir = torch.randint(2).float()*2-1
    #         xc = center_x + xc_dir*box_w*xc_prop
    #         yc_dir = torch.randint(2).float()*2-1
    #         yc = center_y + yc_dir*box_h*yc_prop

    #         for i in range(100): # each center location only try at most 100 times, resort another center location if not
    #             width_min = rand_iou*box_w,
    #             width_max = box_w/rand_iou
    #             width = uniSample(width_min, width_max)

    #             height_min = rand_iou*box_h / min(width, box_w)
    #             height_max = box_h/rand_iou
    #             height = uniSample(height_min, height_max)

    #             rand_box = torch.tensor( [int(xc-width/2), int(yc-height/2), int(xc+width/2), int(yc+height/2)]).float()
    #             rand_box.clamp_(0, image_size-1)
    #             if b_iou(rand_box, box) >= rand_iou:
    #                 break
    #         proposals.append(rand_box)

    #     proposals = torch.stack(proposals, dim=0) # (num, 4) xyxy
    #     return proposals        

    # def handcraft_sample_proposals(self, targets, num_per_image=100):
    #     proposals = []
    #     labels = []

    #     num_sample_per_box = 100
    #     for targets_per_image in targets:
    #         image_size = targets_per_image.size
    #         bboxes_per_image = targets_per_image.bbox
    #         labels_per_image = targets.get_field('labels')
    #         proposals_per_image = []
    #         labels_per_image = []
    #         for bbox, label in zip(bboxes_per_image, labels_per_image):
    #             sampled_bboxes = self.sample_proposal_per_box(bbox, image_size, iou_range=(0.5, 0.9), num=num_sample_per_box)
    #             sampled_labels = label.unsqueeze().repeat(len(sampled_bboxes))
    #             proposals_per_image.append(sampled_bboxes)
    #             labels_per_image.append(sampled_labels)
    #         proposals_per_image = torch.cat(proposals_per_image, dim=0)
    #         labels_per_image = torch.cat(labels_per_image, dim=0)

    #         if len(proposals_per_image) > num_per_image:
    #             inds = torch.randperm(len(proposals_per_image))
    #             selected_inds = inds[:num_per_image]
    #             proposals_per_image = proposals_per_image[selected_inds]
    #             labels_per_image = labels_per_image[selected_inds]

    #         assert len(proposals_per_image) == num_per_image, [ len(proposals_per_image), num_per_image ] 
    #         proposals.append(proposals_per_image)
    #         labels.append(labels_per_image)

    #     return proposals, labels

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        # handcrafted_proposals, handcrafted_labels = self.handcraft_sample_proposals(proposals, targets)
        # assert len(handcrafted_proposals) == len(proposals), [len(handcrafted_proposals), len(proposals)]
        # proposals = [torch.cat([proposals[i], handcrafted_proposals[i]]) for i in range(len(proposals))]

        if self.cfg.FEW_SHOT.SOFT_LABELING:
            labels, soft_labels, regression_targets = self.prepare_targets(proposals, targets)
        else:
            labels, regression_targets = self.prepare_targets(proposals, targets)

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels, neg_supp=self.cfg.FEW_SHOT.NEG_SUPPORT.TURN_ON)
        # print('pos:', len(sampled_pos_inds[0].nonzero()), 'neg', len(sampled_neg_inds[0].nonzero()))
        
        proposals = list(proposals)
        

        # add corresponding label and regression_targets information to the bounding boxes
        if self.cfg.FEW_SHOT.SOFT_LABELING:
            for labels_per_image, soft_labels_per_image, regression_targets_per_image, proposals_per_image in zip(
                labels, soft_labels, regression_targets, proposals
            ):
                proposals_per_image.add_field("labels", labels_per_image)
                proposals_per_image.add_field("soft_labels", soft_labels_per_image)
                proposals_per_image.add_field(
                    "regression_targets", regression_targets_per_image
                )
        else:
            for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
                labels, regression_targets, proposals
            ):
                proposals_per_image.add_field("labels", labels_per_image)
                proposals_per_image.add_field(
                    "regression_targets", regression_targets_per_image
                )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            # assert torch.sum(proposals_per_image.get_field('labels').float() - proposals_per_image.get_field('soft_labels'))==0, \
            #          [proposals_per_image.get_field('labels'), proposals_per_image.get_field('soft_labels')]
            proposals[img_idx] = proposals_per_image


        self._proposals = proposals

        return proposals

    def CXE(self, predicted, target):
        my_target = torch.stack([1-target, target], dim=1)
        return -(my_target * torch.log(predicted)).mean()

    def FOCAL_LOSS(self, predicted_diff):
        '''
            multi cls focal loss
        '''
        EPISILON = 1e-6
        log_pt = torch.log(1-predicted_diff + EPISILON)
        return - ((predicted_diff) * log_pt).mean() 

    def __call__(self, class_logits, box_regression, neg_class_logits=None, rev_class_logits=None, gt_label=-1):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)

        if self.cfg.FEW_SHOT.REVERSE_ORDER:
            assert rev_class_logits is not None
            rev_class_logits = cat(rev_class_logits, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        if self.cfg.FEW_SHOT.SOFT_LABELING:
            soft_labels = cat([proposal.get_field("soft_labels") for proposal in proposals], dim=0)
        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        if gt_label == -1: 
            N = labels.size(0)
            pos_inds = torch.nonzero(labels > 0)
            if self.cfg.FEW_SHOT.SECOND_STAGE_CLS_LOSS == 'focal_loss':
                classification_loss = self.cls_loss_func(
                                class_logits,
                                labels.int()
                            ) / max(pos_inds.numel(), 1)
            elif self.cfg.FEW_SHOT.SECOND_STAGE_CLS_LOSS == 'ce_loss':
                if self.cfg.FEW_SHOT.LOSS_WEIGHTED:
                    fg_weight = 0.75
                    if class_logits.size(1) == 2:
                        weight = torch.tensor([1-fg_weight, fg_weight]).cuda(class_logits.get_device()).float()
                    elif class_logits.size(1) == 3:
                        weight = torch.tensor([1-fg_weight, fg_weight, fg_weight]).cuda(class_logits.get_device()).float()
                    else:
                        raise Exception('class logits dimention wrong, can only be 2 or 3 for softmax ce loss')
                    classification_loss = F.cross_entropy(class_logits, labels, weight=weight) # change to focal loss  
                else:
                    classification_loss = F.cross_entropy(class_logits, labels) # change to focal loss  
            elif self.cfg.FEW_SHOT.SECOND_STAGE_CLS_LOSS == 'mse_loss' and self.cfg.FEW_SHOT.SOFT_LABELING:
                classification_loss = torch.mean((class_logits.sigmoid()-soft_labels)**2)         
            elif self.cfg.FEW_SHOT.SECOND_STAGE_CLS_LOSS == 'mse_loss' and not self.cfg.FEW_SHOT.SOFT_LABELING:
                classification_loss = torch.mean((class_logits.sigmoid()-labels.float())**2)         
            elif self.cfg.FEW_SHOT.SECOND_STAGE_CLS_LOSS == 'l1_loss' and self.cfg.FEW_SHOT.SOFT_LABELING:
                classification_loss = torch.mean(torch.abs(class_logits.sigmoid()-soft_labels))
            elif self.cfg.FEW_SHOT.SECOND_STAGE_CLS_LOSS == 'cxe_loss' and self.cfg.FEW_SHOT.SOFT_LABELING:
                classification_loss = self.CXE(class_logits.softmax(dim=1), soft_labels)
            else:
                raise Exception('clasification loss of second stage not valid')

            if self.cfg.FEW_SHOT.REVERSE_ORDER:
                reverse_cls_loss = self.FOCAL_LOSS(
                    torch.abs(class_logits.softmax(dim=-1)-rev_class_logits.softmax(dim=-1))
                    )

            # get indices that correspond to the regression targets for
            # the corresponding ground truth labels, to be used with
            # advanced indexing
            sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
            labels_pos = labels[sampled_pos_inds_subset]
            if self.cls_agnostic_bbox_reg:
                map_inds = torch.tensor([4, 5, 6, 7], device=device)
            else:
                map_inds = 4 * labels_pos[:, None] + torch.tensor(
                    [0, 1, 2, 3], device=device)

            box_loss = smooth_l1_loss(
                box_regression[sampled_pos_inds_subset[:, None], map_inds],
                regression_targets[sampled_pos_inds_subset],
                size_average=False,
                beta=1,
            )
            box_loss = box_loss / labels.numel()
        else: # only calculating loss of bboxes belonging to gt_label and mse loss
            N = labels.size(0)
            cls_labels = labels.clone()
            cls_labels[torch.nonzero(cls_labels!=gt_label)] = 0
            cls_labels[torch.nonzero(cls_labels==gt_label)] = 1

            pos_inds = torch.nonzero(cls_labels > 0)
            if self.cfg.FEW_SHOT.SECOND_STAGE_CLS_LOSS == 'focal_loss':
                classification_loss = self.cls_loss_func(
                                class_logits,
                                cls_labels.int()
                            ) / ( pos_inds.numel() + N )
            else:
                classification_loss = F.cross_entropy(class_logits, cls_labels) # change to focal loss

            # get indices that correspond to the regression targets for
            # the corresponding ground truth labels, to be used with
            # advanced indexing
            if cls_labels.numel() == 0:
                box_loss = torch.tensor(0).float().cuda(device)
            else:
                sampled_pos_inds_subset = torch.nonzero(cls_labels > 0).squeeze(1)
                labels_pos = cls_labels[sampled_pos_inds_subset]
                if self.cls_agnostic_bbox_reg:
                    map_inds = torch.tensor([4, 5, 6, 7], device=device)
                else:
                    map_inds = 4 * labels_pos[:, None] + torch.tensor(
                        [0, 1, 2, 3], device=device)


                box_loss = smooth_l1_loss(
                    box_regression[sampled_pos_inds_subset[:, None], map_inds],
                    regression_targets[sampled_pos_inds_subset],
                    size_average=False,
                    beta=1,
                )
                box_loss = box_loss / cls_labels.numel()
        if self.cfg.FEW_SHOT.REVERSE_ORDER:
            return classification_loss, box_loss, reverse_cls_loss

        # add in new neg support linz
        if neg_class_logits is not None: 
            neg_class_logits = cat(neg_class_logits, dim=0)
            focus_neg_class_logits = neg_class_logits[labels==1]
            focus_pos_class_logits = class_logits[labels==1]
            # focus_labels = labels[labels==1]
            focus_neg_class_scores = focus_neg_class_logits.softmax(dim=1)[:,1]
            focus_pos_class_scores = focus_pos_class_logits.softmax(dim=1)[:,1]
            cls_suppress_loss = F.relu(focus_neg_class_scores-focus_pos_class_scores+0.3).mean()
            # cls_suppress_loss = F.cross_entropy(focus_neg_class_logits, 1-focus_labels) # change to focal loss  
            return classification_loss, box_loss, cls_suppress_loss

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        cfg,
        matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
