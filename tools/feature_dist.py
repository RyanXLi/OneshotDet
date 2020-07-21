import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import os
import sys

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))

from maskrcnn_benchmark.utils.env import setup_environment
from maskrcnn_benchmark.data.datasets.coco import COCODataset
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.image_list import to_image_list

print(os.getcwd())
os.chdir('..')

cfg.merge_from_file('/data/xlide/fcos/configs/fcos/feature_dist_conf.yaml')
cfg.freeze()

# transforms = build_transforms(cfg, is_train=False)
dataloader = make_data_loader(cfg, is_train=False)


resnet50 = models.resnet50(pretrained=True)
modules = list(resnet50.children())[:-1]
resnet50 = nn.Sequential(*modules)
for p in resnet50.parameters():
    p.requires_grad = False
resnet50.eval()
device = 'cuda'
resnet50.to(device)

for step, samples in enumerate(dataloader):
    for sample in samples:
        images, images_support, images_neg_support, targets, img_ids = sample
        images = images.to(device)
        images_support = images_support.to(device)
        images_neg_support = images_neg_support.to(device)
        targets = [target.to(device) for target in targets]
        images = to_image_list(images)
        images_support = to_image_list(images_support)
        query_images_tensor = images.tensors
        supp_images_tensor = images_support.tensors

        # img = torch.Tensor(1, 3, 224, 224).normal_()
        features = resnet50(query_images_tensor)

        features_supp = resnet50(supp_images_tensor)
        # img = torch.Tensor(3, 224, 224).normal_() # random image
        # img_var = Variable(img) # assign it to a variable
        # features_var = resnet50(img_var) # get the output from the last hidden layer of the pretrained resnet
        # features = features_var.data # get the tensor out of the variable
        print(img_ids)
        print(features.shape)
        # print(features)

        print(features_supp.shape)
        # print(features_supp)

        print()