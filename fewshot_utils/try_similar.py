import torch
import torchvision
import torch.nn as nn
import torchvision.models as models

import random
from PIL import Image
import os
import time
import numpy as np

import matplotlib
matplotlib.use('Agg') #  disable display
import matplotlib.pyplot as plt
import shutil
import time

def cos_dist(a, b):
    a_norm = a / a.norm(dim=0)
    b_norm = b / b.norm(dim=0)
    return torch.mm(a_norm, b_norm.transpose(0,1))

def find_similar(path):
    cat_dir, name = os.path.split(path)
    _, cat = os.path.split(cat_dir)
    img_path = path.replace('supps_feat', 'supps').replace('pth', 'jpg')
    feat = torch.load(path)
    img_cat_dir = cat_dir.replace('supps_feat', 'supps')
    candidates = []
    for item in os.listdir(cat_dir):
        if item == name:
            continue
        feat_supp = torch.load(os.path.join(cat_dir, item))
        sim = nn.functional.cosine_similarity(feat['x3'].to('cuda'), feat_supp['x3'].to('cuda'))
        candidates.append((item, sim.to('cpu').item()))

    candidates.sort(key=lambda x: x[1], reverse=True)   



def find_similar_loaded(path, feats):
    cat_dir, name = os.path.split(path)
    _, cat = os.path.split(cat_dir)
    img_path = path.replace('supps_feat', 'supps').replace('pth', 'jpg')
    feat = feats[name]
    img_cat_dir = cat_dir.replace('supps_feat', 'supps')
    candidates = []
    for item in os.listdir(cat_dir):
        if item == name:
            continue
        feat_supp = feats[item]#torch.load(os.path.join(cat_dir, item))
        sim = nn.functional.cosine_similarity(feat['x3'].to('cuda'), feat_supp['x3'].to('cuda'))
        candidates.append((item, sim.to('cpu').item()))
    
    candidates.sort(key=lambda x: x[1], reverse=True)   



def find_similar_for_cat(cat):
    base_dir = ''
    img_cat_dir = base_dir + 'supps/' + cat + '/'
    feat_cat_dir = base_dir + 'supps_feat/' + cat + '/'
    img_names = os.listdir(img_cat_dir)

    feat_supp = torch.load(os.path.join(feat_cat_dir, item))

    candidates = []
    for item in os.listdir(cat_dir):
        if item == name:
            continue
        feat_supp = torch.load(os.path.join(cat_dir, item))
        sim = nn.functional.cosine_similarity(feat['x3'].to('cuda'), feat_supp['x3'].to('cuda'))
        candidates.append((item, sim.to('cpu').item()))

    candidates.sort(key=lambda x: x[1], reverse=True)   


tic = time.time()
feats = {}
cat_dir = 'supps_feat/49/'
for item in os.listdir(cat_dir):
    feats[item] = torch.load(os.path.join(cat_dir, item))
toc = time.time()
print('loading time: ' + str(toc - tic))

tic = time.time()
find_similar_loaded('supps_feat/49/113_696866.pth', feats)
toc = time.time()
print('calc1 time: ' + str(toc - tic))

tic = time.time()
find_similar('supps_feat/49/113_696866.pth')
toc = time.time()
print('calc orig time: ' + str(toc - tic))