# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms, boxlist_soft_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        cfg,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=10000,
        box_coder=None,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.cfg=cfg
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def forward(self, x, boxes, cyclic=False, target_ids=None):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        # proposals = boxes
        box_regression = box_regression[:, :8] #         add linz
        if self.cfg.FEW_SHOT.SECOND_STAGE_CLS_LOSS == 'focal_loss':
            class_prob = class_logits.sigmoid()[:, :1]
            bg_prob = 1-class_prob
            class_prob = torch.cat([bg_prob, class_prob], dim=1)
        elif self.cfg.FEW_SHOT.SECOND_STAGE_CLS_LOSS in ['ce_loss', 'cxe_loss']:
            class_prob = F.softmax(class_logits, -1)[:, :2]
        elif self.cfg.FEW_SHOT.SECOND_STAGE_CLS_LOSS in ['mse_loss','l1_loss']:
            class_logits = class_logits.sigmoid()
            class_prob = torch.cat([1-class_logits, class_logits], dim=1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]

        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )

        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = 2

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)
        if cyclic:
            class_prob_ind = [class_prob_per_img[:,1].argmax() for class_prob_per_img in class_prob]
            class_prob = [class_prob_per_image[ind] for class_prob_per_img, ind in zip(class_prob, class_prob_ind)]
            proposals = [ proposals[ind] for proposals_per_img, ind in zip(proposals, class_prob_ind) ]

        results = []
        for prob, boxes_per_img, image_shape, target_id in zip(
            class_prob, proposals, image_shapes, target_ids
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape, target_id=target_id)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist, num_classes, target_id)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape, cyclic=False, target_id=None):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4) # n*num_cls, 4
        scores = scores.reshape(-1)  # n*num_cls
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        if cyclic:
            labels = torch.zeros_like(scores).long().fill_(target_id)
            boxlist.add_field("labels", labels)
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes, target_id=None):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )


            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), target_id, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            sorted_cls_scores, sorted_cls_indices = torch.sort(cls_scores, descending=True)
            keep = sorted_cls_indices[:self.detections_per_img]
            result = result[keep]
        return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    postprocessor = PostProcessor(
        cfg,
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg
    )
    return postprocessor
