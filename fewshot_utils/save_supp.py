import random
from PIL import Image
import os
import time
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import multiprocessing

min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def save_one_cat(cat):
    if cat == None:
        return
    print('starting for '+str(cat))
    for img_id in catalog[cat]:
        img_meta_info = coco.loadImgs(img_id)[0]
        path = img_meta_info['file_name']

        img = Image.open(os.path.join(root, path)).convert('RGB')

        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat)
        targets = coco.loadAnns(ann_ids)

        coords = [target['bbox'] for target in targets]
        w, h = img.size
        for i in range(len(coords)):
            coord = coords[i]
            if coord[0]+coord[2] > w or coord[1]+coord[3] > h:
                print('oversized box for image '+str(img_id))
                continue
            img_cropped = img.crop((coord[0], coord[1], coord[0]+coord[2], coord[1]+coord[3]))
            save_dir = os.path.join(dirName, str(cat))
            save_dir = save_dir + '/' + str(img_id) + '_' + str(ann_ids[i]) + '.jpg'
            try:
                img_cropped.save(save_dir)
            except Exception:
                print('oversized box for image '+str(img_id) + ' Annotation: '+str(ann_ids[i]))
                continue

    print('finished for '+str(cat))


coco_json_dir = "datasets/voc/COCOAnnotations/pascal_test2007.json"
root = 'datasets/voc/VOC2007/JPEGImages'

# coco_json_dir = "coco/annotations/instances_train2017.json"
# root = "coco/train2017"

coco = COCO(coco_json_dir)


# train-test split
voc_class = [1, 2, 3, 4, 5, 6, 7, 9, 40, 57, 59, 61, 63, 15, 16, 17, 18, 19, 20] # in terms of 1-80
exclude_cont_catIds = []#[80, 78, 13, 34, 55, 75, 33, 70, 48, 77, 28, 63, 47, 14, 23, 11, 13, 36, 34, 35, 24, 22, 32, 16, 9, 20, 10, 25, 19, 39, 28, 4, 33, 17, 7, 6, 21, 55, 31, 57, 60, 40, 8, 56, 54, 2, 5, 38, 80, 74, 78, 76, 53, 37, 41, 59, 89, 75, 27, 46, 87, 90, 79, 18, 82, 44, 51, 77, 52, 48, 81, 15, 88, 86, 61, 49, 70, 73, 84, 72, 85, 64, 63, 65, 47, 3, 67]#voc_class
remove_images_without_annotations = True

# printing excluded classes
all_json_category_id_to_contiguous_id = {
    v: i + 1 for i, v in enumerate(coco.getCatIds())
} # 1-90 to 1-80
all_contiguous_category_id_to_json_id = {
    v: k for k, v in all_json_category_id_to_contiguous_id.items()
} # 1-80 to 1-90

json_category_id_to_contiguous_id = {
    v: i + 1 for i, v in enumerate(coco.getCatIds()) if v not in exclude_cont_catIds # 1-90 to 1-80 included cls
}
contiguous_category_id_to_json_id = {
    v: k for k, v in json_category_id_to_contiguous_id.items()
}

json_cat_list = list(json_category_id_to_contiguous_id.keys()) # 1-90
catalog = {} # cat to list of img_ids
for cat in json_cat_list:
    catalog[cat] = []
    img_ids_cur_cat = coco.getImgIds(catIds=cat)
    img_ids_cur_cat = sorted(img_ids_cur_cat)
    # filter images without detection annotations
    if remove_images_without_annotations:
        for img_id in img_ids_cur_cat:
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False) # filter out iscrowd
            anno = coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                catalog[cat].append(img_id)

timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
dirName = 'supps_test_voc2007/'
if not os.path.exists(dirName):
    os.makedirs(dirName)


for cat in json_cat_list:
    if not os.path.exists(dirName+str(cat)):
        os.makedirs(dirName+str(cat))

sorted(json_cat_list)
num_cores = 32

with multiprocessing.Pool(num_cores) as p:
    p.map(save_one_cat, json_cat_list)
    