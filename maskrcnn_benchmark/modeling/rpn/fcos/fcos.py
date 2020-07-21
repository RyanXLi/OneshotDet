import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from maskrcnn_benchmark.layers import Scale


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()

        num_classes = 1

        self.cfg=cfg
        self.in_channels=in_channels
        cls_tower = []
        bbox_tower = []

        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.dense_points = cfg.MODEL.FCOS.DENSE_POINTS
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4 * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1 * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x, is_neg_supp=False):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            # check debug
            cls_tower = self.cls_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(cls_tower))# lin change centerness to bbox tower

            bbox_tower = self.bbox_tower(feature) 
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(bbox_tower)
            )))

        return logits, bbox_reg, centerness


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()
        self.cfg=cfg
        self.head = FCOSHead(cfg, in_channels)
        if not cfg.MODEL.RPN_ONLY:
            self.box_selector_train = make_fcos_postprocessor(cfg, is_train=True)
        self.box_selector_test = make_fcos_postprocessor(cfg, is_train=False)
        self.loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.dense_points = cfg.MODEL.FCOS.DENSE_POINTS

    def neg_cls_loss(self, box_pos_cls, box_neg_cls):
        '''
            lin for fun return loss = -p*log(0.5*(p-n+1))
        '''
        loss_dict = {}
        loss = 0
        for i in range(len(box_pos_cls)):
            box_pos_sigmoid = F.sigmoid(box_pos_cls[i])
            box_neg_sigmoid = F.sigmoid(box_neg_cls[i])
            loss+= torch.mean(- box_pos_sigmoid * torch.log(0.5*(box_pos_sigmoid-box_neg_sigmoid+1)))
        loss/=len(box_pos_cls)
        loss_dict['neg_supp_loss'] = loss
        return loss_dict

    def clean_targets(self, targets):
        cleaned_targets = []
        for i in range(len(targets)):
            target = targets[i]
            labels = target.get_field('labels')
            valid_pos_ind = [ t for t in range(len(labels)) if labels[t]==1 ]
            new_target = target.copy_with_fields(target.fields())
            new_target.bbox = new_target.bbox[valid_pos_ind]
            new_target.add_field('labels', new_target.get_field('labels')[valid_pos_ind])
            cleaned_targets.append(new_target)
        return cleaned_targets

    def forward(self, images, features, targets=None, neg_features=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)

        locations = self.compute_locations(features)

        if self.training:
            cleaned_targets = self.clean_targets(targets)
            temp, losses = self._forward_train(
                locations, box_cls,
                box_regression,
                centerness, cleaned_targets, images.image_sizes
            )
            return temp, losses
        else:
            return self._forward_test(
                locations, box_cls, box_regression,
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets, image_sizes):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = locations
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(
                    locations, box_cls, box_regression, centerness, image_sizes, targets
                )
        return boxes, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression,
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        locations = self.get_dense_locations(locations, stride, device)
        return locations

    def get_dense_locations(self, locations, stride, device):
        if self.dense_points <= 1:
            return locations
        center = 0
        step = stride // 4
        l_t = [center - step, center - step]
        r_t = [center + step, center - step]
        l_b = [center - step, center + step]
        r_b = [center + step, center + step]
        if self.dense_points == 4:
            points = torch.cuda.FloatTensor([l_t, r_t, l_b, r_b], device=device)
        elif self.dense_points == 5:
            points = torch.cuda.FloatTensor([l_t, r_t, [center, center], l_b, r_b], device=device)
        else:
            print("dense points only support 1, 4, 5")
        points.reshape(1, -1, 2)
        locations = locations.reshape(-1, 1, 2).to(points)
        dense_locations = points + locations
        dense_locations = dense_locations.view(-1, 2)
        return dense_locations


def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
