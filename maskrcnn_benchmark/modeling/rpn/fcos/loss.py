"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn

from ..utils import concat_box_prediction_layers
from maskrcnn_benchmark.layers import IOULoss
from maskrcnn_benchmark.layers import SigmoidFocalLoss, SoftmaxFocalLoss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

import os
import time

INF = 100000000


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        if cfg.LOSS.FOCAL_LOSS == 'SIGMOID':
            self.cls_loss_func = SigmoidFocalLoss(
                cfg.MODEL.FCOS.LOSS_GAMMA,
                cfg.MODEL.FCOS.LOSS_ALPHA
            )
        elif cfg.LOSS.FOCAL_LOSS == 'SOFTMAX':
            self.cls_loss_func = SoftmaxFocalLoss(
                cfg.MODEL.FCOS.LOSS_GAMMA,
                cfg.MODEL.FCOS.LOSS_ALPHA
            )
        else:
            raise Exception('no focal loss type named {}'.format(cfg.LOSS.FOCAL_LOSS))
        self.center_sample = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.radius = cfg.MODEL.FCOS.POS_RADIUS
        self.loc_loss_type = cfg.MODEL.FCOS.LOC_LOSS_TYPE
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.loc_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.dense_points = cfg.MODEL.FCOS.DENSE_POINTS

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1):
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        try:
            # no gt
            if len(center_x.size()) == 0 or center_x[..., 0].sum() == 0: # ADDED or
                return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        except Exception:
            timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            dirName = 'errors/' + timeStamp + '/'
            if not os.path.exists(dirName):
                os.makedirs(dirName)
            with open(dirName+'err.txt', 'a') as f:
                f.write(str(center_x))
                f.write(str(center_x.size()))
                f.write(str(center_y))
                f.write(str(center_y.size()))
                f.write('\n\n')

            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)



        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1) # (n_pts, 2)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0) # (5*n_pts, 2)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0) # (num_pts_per_level*5, ...)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        ) # (n_imgs, 5*pts)  (n_imgs, 5*pts, 4)

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)  # (5, pts)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0) # (5, pts, 4)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)): # 5
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0) # (n_imgs*pts)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0) # (n_imgs*pts, 4)
            )

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        ''' 
            input location: (pts*5, 4) x1, y1, x2, y2
            targets: list of image targets
            obj_si... :  (pts*5, 2)
        '''
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)): # for each image
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"


            ###############################################################
            ########################## compute labels      ################
            ###############################################################
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2) # (5*pts, n_targets, 4)
            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.strides,
                    self.num_points_per_level,
                    xs,
                    ys,
                    radius=self.radius)
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            # print('reg size', reg_targets_per_im.size())
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1) # (5*pts, n_targets)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds] # (5*pts, 4)
            labels_per_im = labels_per_im[locations_to_gt_inds] # (5*pts)
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            # loss_isnan = torch.isnan(labels_per_im)
            # assert torch.sum(loss_isnan) == 0, ['labels_per_im', labels_per_im]


        return labels, reg_targets # (n_imgs, 5*pts)  (n_imgs, 5*pts, 4)

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])              (N, C)
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1) // self.dense_points

        labels, reg_targets = self.prepare_targets(locations, targets) # (n_imgs*pts), (n_imgs*pts, 4)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].permute(0, 2, 3, 1).reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0) # (n_imgs*pts*bs*h*w, c)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0) # (n_imgs*pts*bs*h*w, 4)
        centerness_flatten = torch.cat(centerness_flatten, dim=0) # (n_imgs*pts*bs*h*w)
        labels_flatten = torch.cat(labels_flatten, dim=0) # (n_imgs*pts)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0) # (n_imgs*pts, 4)
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1) # (n_imgs*pts)

        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        box_regression_flatten = box_regression_flatten[pos_inds]

        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets,
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
