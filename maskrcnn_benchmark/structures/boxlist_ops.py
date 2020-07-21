# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from .bounding_box import BoxList

from maskrcnn_benchmark.layers import nms as _box_nms
import numpy as np


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    # print('box', boxes.size())
    # assert boxes.size(0) == score.size(0), [boxes.size(), score.size()]
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)

def boxlist_soft_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)

    # print('bb box',boxes.size())
    # print('bb score', score.size())
    # print('box', boxes.size())
    # print('score', score.size())
    # keep = py_cpu_nms(boxes.cpu().detach().numpy(), score.cpu().detach().numpy(), nms_thresh)
    keep = py_gpu_nms(boxes, score, nms_thresh)
    keep = torch.tensor(keep).long().cuda(boxes.get_device())

    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]

    # boxlist.bbox = box_keep
    # boxlist.add_field(score_field, scores_keep)

    return boxlist.convert(mode)

def py_cpu_nms(boxes, scores, thresh=0.3):
    """Pure Python NMS baseline."""
    thresh = 0.3
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def py_gpu_nms(boxes, scores, thresh=0.3):
    """Pure Python NMS baseline."""
    thresh = 0.3
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort(dim=-1, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        xx1 = torch.max(x1[i:i+1], x1[order[1:]])
        yy1 = torch.max(y1[i:i+1], y1[order[1:]])
        xx2 = torch.max(x2[i:i+1], x2[order[1:]])
        yy2 = torch.max(y2[i:i+1], y2[order[1:]])

        w = F.relu(xx2 - xx1 + 1)
        h = F.relu(yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # inds = np.where(ovr <= thresh)[0]
        inds = torch.nonzero(ovr<=thresh)[:,0]
        order = order[inds + 1]

    return keep

def box_soft_nms(bboxes, scores, nms_threshold=0.3, max_proposals=2000000, soft_threshold=0.001, sigma=0.5,mode='union'):
    """
    soft-nms implentation according the soft-nms paper
    :param bboxes:
    :param scores:
    :param labels:
    :param nms_threshold:
    :param soft_threshold:
    :return:
    """
    nms_threshold = 0.3
    box_keep = []
    labels_keep = []
    scores_keep = []
    c = 1
    c_boxes = bboxes
    c_scores = scores
    weights = c_scores.clone()
    x1 = c_boxes[:, 0]
    y1 = c_boxes[:, 1]
    x2 = c_boxes[:, 2]
    y2 = c_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = weights.sort(0, descending=True)
    while order.numel() > 0:
        if len(order.size()) == 0:
            break
        # print(order.size(), order.numel())
        # if order.numel() == 1:
        #     print(order)
        i = order[0]
        box_keep.append(c_boxes[i])
        scores_keep.append(c_scores[i])
        if len(box_keep) > max_proposals:
            break

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids_t= (ovr>=nms_threshold).nonzero().squeeze()

        weights[[order[ids_t+1]]] *= torch.exp(-(ovr[ids_t] * ovr[ids_t]) / sigma) # gaussian

        ids = (weights[order[1:]] >= soft_threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        c_boxes = c_boxes[order[1:]][ids]
        c_scores = weights[order[1:]][ids]
        _, order = weights[order[1:]][ids].sort(0, descending=True)
        if c_boxes.dim()==1:
            c_boxes=c_boxes.unsqueeze(0)
            c_scores=c_scores.unsqueeze(0)
        x1 = c_boxes[:, 0]
        y1 = c_boxes[:, 1]
        x2 = c_boxes[:, 2]
        y2 = c_boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    return torch.stack(box_keep), torch.stack(scores_keep)



def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())

    if not [set(bbox.fields()) for bbox in bboxes]:
        assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
