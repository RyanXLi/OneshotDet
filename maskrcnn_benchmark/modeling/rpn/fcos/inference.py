import torch
import copy
import numpy as np
import random
import multiprocessing

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms, boxlist_soft_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(self, config, pre_nms_thresh, pre_nms_top_n, nms_thresh,
                 fpn_post_nms_top_n, min_size, num_classes, dense_points, score_calculator):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.cfg=config
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.dense_points = dense_points
        self.score_calculator = score_calculator

    def forward_for_single_feature_map(self, locations, box_cls,box_regression, centerness,image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        if self.score_calculator == 'BINARY':
            box_cls = box_cls.reshape(N, -1, self.num_classes - 1).sigmoid()
            if self.num_classes > 2: # having neg support classes
                pos_cls = box_cls[:,:,0:1]
                box_cls = pos_cls

        elif self.score_calculator == 'MULTI':
            box_cls = box_cls.reshape(N, -1, self.num_classes)[:,:,:2].softmax(dim=2)
            box_cls = box_cls[:,:,1:2]
        else:
            raise Exception('loss type wrong')

        box_regression = box_regression.view(N, self.dense_points * 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, self.dense_points, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()

        candidate_inds = box_cls > self.pre_nms_thresh # N, h*w, 1
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i] # (h*w, 1)
            per_candidate_inds = candidate_inds[i] # (h*w, 1)
            per_box_cls = per_box_cls[per_candidate_inds] # (h*w, 1, 1)

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1 # cls+1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist.add_field("objectness", per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            # if not self.cfg.MODEL.RPN_ONLY:
            #     if self.cfg.FEW_SHOT.NMS=='nms':
            #         boxlist = boxlist_nms(
            #             boxlist,
            #             self.nms_thresh,
            #             max_proposals=self.fpn_post_nms_top_n,
            #             score_field="objectness",
            #         )
            #     else:
            #         # print('b ', boxlist.bbox.size())
            #         boxlist = boxlist_soft_nms(
            #             boxlist,
            #             self.nms_thresh,
            #             max_proposals=self.fpn_post_nms_top_n,
            #             score_field="objectness",
            #         )
                    # print('a ', boxlist.bbox.size())
            results.append(boxlist)

        return results

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("scores", torch.ones(len(gt_box), device=device))

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

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
            # print('iou', iou.item())
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
                # print(box)
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
                        # print('bin_idx', bin_idx)
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
            # gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))
            gt_box.add_field("scores", torch.ones(len(gt_box), device=device))
            # gt_box.add_field("labels", torch.ones(len(gt_box), device=device).long())

        proposals = [
            cat_boxlist((gt_box, proposal))
            for proposal, gt_box in zip(proposals, artificial_boxes)
        ]

        proposals = [proposal[:self.fpn_post_nms_top_n] for proposal in proposals] 

        return proposals

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, image_sizes
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)
        if not self.cfg.MODEL.RPN_ONLY:
            if self.training and targets is not None:
                # remove negative support target
                if self.cfg.FEW_SHOT.ADD_ARTIFICIAL_PROPOSALS:
                    boxlists = self.add_artificial_proposals(boxlists, targets)
                else:
                    boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists



    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []

            scores_j = scores#[inds]
            boxes_j = boxes#[inds, :].view(-1, 4)
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)

            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms_thresh,
                score_field="scores"
            )

            num_labels = len(boxlist_for_class)

            result.append(boxlist_for_class)

            result = cat_boxlist(result)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                sorted_cls_scores, sorted_cls_indices = torch.sort(cls_scores, descending=True)
                keep = sorted_cls_indices[:self.fpn_post_nms_top_n]

                result = result[keep]
            results.append(result)
        return results

def make_fcos_postprocessor(config, is_train):
    if config.MODEL.RPN_ONLY:
        pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH # 0.05
        pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N # 1000
        nms_thresh = config.MODEL.FCOS.NMS_TH # 0.6
        fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG # 100
        dense_points = config.MODEL.FCOS.DENSE_POINTS # 1
        num_cls = config.MODEL.FCOS.NUM_CLASSES
        if config.FEW_SHOT.NEG_SUPPORT.TURN_ON:
            num_cls += config.FEW_SHOT.NEG_SUPPORT.NUM_CLS
        score_calculator = config.LOSS.CLS_LOSS
        min_size = 0
    else:
        num_cls = 2
        pre_nms_thresh = 0
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN # 2000
        if not is_train:
            fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST # 1000
        dense_points = config.MODEL.FCOS.DENSE_POINTS # 1
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN # 2000
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN # 2000
        if not is_train:
            pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST # 2000
            post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST # 1000
        nms_thresh = config.MODEL.RPN.NMS_THRESH # 0.8
        min_size = config.MODEL.RPN.MIN_SIZE # 0
        score_calculator = config.LOSS.CLS_LOSS

    box_selector = FCOSPostProcessor(
        config=config,
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=min_size,
        num_classes=num_cls,
        dense_points=dense_points,
        score_calculator=score_calculator)

    return box_selector