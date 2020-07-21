# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from maskrcnn_benchmark.utils.comm import get_rank

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
# from one_shot_supp import one_shot_supp_list

import random
from PIL import Image
import os
import pickle
import time
import numpy as np
import pickle

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

def _has_only_small_bbox(anno):
    return all(obj['area']<32*32 for obj in anno)

def has_valid_large_annotation(anno):
    if not has_valid_annotation(anno):
        return False
    return not _has_only_small_bbox(anno)


class ImagenetDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, 
        cfg,
        ann_file, 
        root, 
        is_train, 
        remove_images_without_annotations, 
        transforms=None, 
        save_img=False, 
    ):
        super(ImagenetDataset, self).__init__(root, ann_file)
        self.rank = get_rank()
        # random.seed(6666)
        remove_images_without_annotations = True

        self.cfg                    = cfg
        self.neg_supp               = cfg.FEW_SHOT.NEG_SUPPORT.TURN_ON
        self.neg_supp_num_cls       = cfg.FEW_SHOT.NEG_SUPPORT.NUM_CLS
        self.choose_close           = False #cfg.FEW_SHOT.CHOOSE_CLOSE
        self.shot                   = cfg.FEW_SHOT.NUM_SHOT
        self.save_img               = cfg.FEW_SHOT.SAVE_IMAGE
        self.training_exclude_cats  = cfg.FEW_SHOT.TRAINING_EXCL_CATS
        self.test_exclude_cats      = cfg.FEW_SHOT.TEST_EXCL_CATS
        self.is_train               = is_train

        if self.rank == 0:
            print('Save image: '+str(self.save_img))
            print('number of shots: '+str(self.shot))
            print('Use negative supp: '+str(self.neg_supp))
            print('Choose close supp: '+str(self.choose_close))

        # train-test split
        voc_class = [1, 2, 3, 4, 5, 6, 7, 9, 40, 57, 59, 61, 63, 15, 16, 17, 18, 19, 20] # in terms of 1-80
        # exclude_cont_catIds = voc_class
        if is_train:
            exclude_cont_catIds = self.training_exclude_cats
        else:
            exclude_cont_catIds = self.test_exclude_cats

        # printing excluded classes
        self.all_json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        } # 1-90 to 1-80
        self.all_contiguous_category_id_to_json_id = {
            v: k for k, v in self.all_json_category_id_to_contiguous_id.items()
        } # 1-80 to 1-90
        is_train_str = 'training' if is_train else 'testing'
        real_excluded_class = [self.all_contiguous_category_id_to_json_id[catId] for catId in exclude_cont_catIds] # to 1-90
        cats = self.coco.loadCats(real_excluded_class)
        names=[cat['name'] for cat in cats]

        if self.rank == 0:
            print('Init COCO for ' + is_train_str + '...')
            print('excluding:')
            print(real_excluded_class)
            print(names)

        # self.json_category_id_to_contiguous_id = {
        #     v: i + 1 for i, v in enumerate(self.coco.getCatIds()) if v in [11] # 1-90 to 1-80 included cls
        # }

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds()) if i + 1 not in exclude_cont_catIds # 1-90 to 1-80 included cls
        }

        if self.rank == 0:
            print('number of ' + is_train_str + ' categories: ' + str(len(self.json_category_id_to_contiguous_id)))
        
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        if isinstance(transforms, list): # which must be true for few shot learning
            self._transforms = transforms[0]
            self._supp_transforms = transforms[1]
        else:
            raise Exception('require a list of 2 _transforms for supp and query')

        self.json_cat_list = list(self.json_category_id_to_contiguous_id.keys()) # 1-80
        self.catalog = {} # cat to list of img_ids
        for cat in self.json_cat_list:
            self.catalog[cat] = []
            img_ids_cur_cat = self.coco.getImgIds(catIds=cat)
            img_ids_cur_cat = sorted(img_ids_cur_cat)
            # filter images without detection annotations
            if remove_images_without_annotations:
                for img_id in img_ids_cur_cat:
                    ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat, iscrowd=False) # filter out iscrowd
                    anno = self.coco.loadAnns(ann_ids)
                    if has_valid_large_annotation(anno):
                        self.catalog[cat].append(img_id)
                        # constrain each cat to only have at most 2000 images
                        # prevent overfitting on cats having more images
                        if len(self.catalog[cat]) >= 2000:
                            break

        # print #imgs per cat
        # if self.rank == 0:
        #     for cat in self.catalog:
        #         loaded_cat = self.coco.loadCats(cat)
        #         name=loaded_cat[0]['name']
        #         print(str(cat) + '(' + name + '): ' + str(len(self.catalog[cat])))

        # # sort indices for reproducible results
        self.ids = []
        self.chosen_cats = []
        self.chosen_neg_cats = []
        for cat, ids in self.catalog.items(): # cat to image ids
            cats = [cat for i in range(len(ids))]
            neg_cats = [-1 for i in range(len(ids))]
            self.chosen_cats = self.chosen_cats + cats
            self.ids = self.ids + ids
            self.chosen_neg_cats = self.chosen_neg_cats + neg_cats
        assert(len(self.ids) == len(self.chosen_cats))
        assert(len(self.ids) == len(self.chosen_neg_cats))

        index_arr = list(range(len(self.ids)))
        random.shuffle(index_arr)
        index_arr = np.asarray(index_arr)
        self.ids = np.asarray(self.ids)
        self.ids = self.ids[index_arr]
        self.ids = self.ids.tolist()
        self.chosen_cats = np.asarray(self.chosen_cats)[index_arr].tolist() # 0->1000 --> cat

    

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
   
        debug_check = False
        if debug_check:
            for i in range(len(self.ids)):
                img_id = self.id_to_img_map[i]
                cur_cat = self.chosen_cats[i]

                ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cur_cat, iscrowd=False) # filter out iscrowd
                anno = self.coco.loadAnns(ann_ids)
                if not anno:
                    print('ERROR: no gt')
                    print(i)
                    print(img_id)
                    print(cur_cat)
                loaded_img = self.coco.loadImgs([img_id])
                if not loaded_img:
                    print('no_img')
                    print(i)
                    print(img_id)
                    print(cur_cat)
                # path = loaded_img[0]['file_name']
        
        # for saving img for vis
        self.dirName = None
        if self.save_img:
            suffix = str(self.shot) + 'shot'
            if self.neg_supp:
                suffix = suffix + '_neg'
            if self.choose_close:
                suffix = suffix + '_close'
            timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            dirName = 'imgs/dist_' + suffix + timeStamp + '/'
            if not os.path.exists(dirName):
                os.makedirs(dirName)
            self.dirName = dirName

        if self.rank == 0:
            print('dataset length: ' + str(self.__len__()))

        # load close dict pkl
        if self.choose_close:
            if self.is_train:
                self.close_dict = {}
                for cat in self.json_cat_list:
                    close_dict_path = '/data/linz/few_shot/fcos_plus/fcos_plus/supp_sim/supp_similarity_'+str(cat)+'.pkl'
                    with open(close_dict_path, 'rb') as f:
                        self.close_dict[cat] = pickle.load(f)
                print('loaded all similarity')
                self.supp_root = '/data/xlide/fcos/supps'
            else: # test time load voc stuff
                pass
        
        cats = self.coco.loadCats(self.json_cat_list)
        names=[cat['name'] for cat in cats]
        print(names)

    def get_one_preset_item_from_cat(self, catId):
        """
        Args:
            catId (int): json cat id

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        assert(catId in self.json_cat_list)
        catId -= 1 # 1-20 to 0-19
        coco = self.coco

        imgs_choices = self.catalog[catId].copy()
        random.shuffle(imgs_choices)

        img_meta_infos = coco.loadImgs(valid_img_ids)
        paths = [img_meta_info['file_name'] for img_meta_info in img_meta_infos]

        imgs = [Image.open(os.path.join(self.root, path)).convert('RGB') for path in paths]
        coords = [chosen_anno['bbox'] for chosen_anno in valid_anns]
        imgs = [imgs[i].crop((coords[i][0], coords[i][1], coords[i][0]+coords[i][2], coords[i][1]+coords[i][3])) for i in range(shot)]

        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]

        return imgs

    def get_random_item_from_cat(self, catId, excludeImgId, shot=1):
        """
        Args:
            catId (int): json cat id

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        assert(catId in self.json_cat_list)

        imgs_choices = self.catalog[catId].copy()
        random.shuffle(imgs_choices)


        pp = False
        valid_img_ids = []
        valid_anns = []
        for img_id in imgs_choices:
            if img_id == excludeImgId:
                continue
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=catId, iscrowd=False)
            target = coco.loadAnns(ann_ids)
            chosen_anno = target[0]
            for anno in target:
                if anno['area'] > chosen_anno['area']:
                    chosen_anno = anno
            pp = chosen_anno['area'] > self.cfg.INPUT.SUPP_AREA_THRESHOLD
            if pp:
                valid_img_ids.append(img_id)
                valid_anns.append(chosen_anno)
                pp=False
            if len(valid_img_ids) == shot:
                break  

        img_meta_infos = coco.loadImgs(valid_img_ids)
        paths = [img_meta_info['file_name'] for img_meta_info in img_meta_infos]

        imgs = [Image.open(os.path.join(self.root, path)).convert('RGB') for path in paths]
        coords = [chosen_anno['bbox'] for chosen_anno in valid_anns]
        imgs = [imgs[i].crop((coords[i][0], coords[i][1], coords[i][0]+coords[i][2], coords[i][1]+coords[i][3])) for i in range(shot)]

        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]

        return imgs

    def get_close_item_from_cat(self, queryImgId, catId, shot=1):
        """
        Args:
            catId (int): json cat id
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        if not self.is_train:
            # print('Not training, cannot get close item!')
            return self.get_random_item_from_cat(catId, queryImgId, shot=self.shot)

        try:
            annDict = self.close_dict[catId][queryImgId][catId] 
        except Exception:
            print(queryImgId)
            print(catId)
            print(self.close_dict[catId][queryImgId])
            return self.get_random_item_from_cat(catId, queryImgId, shot=self.shot)
        
        annList = []
        for len100dict in annDict.values(): # value is a len100 dict
            len100list = [(supp_name, sim) for supp_name, sim in len100dict.items()]
            len100list.sort(key=lambda x: x[0])
            annList.append(len100list)

        num_avail_supp = len(annList[0])
        reduced_annList = [[annList[0][j][0], 0] for j in range(num_avail_supp)]
        for i in range(len(annList)):
            for j in range(num_avail_supp):
                reduced_annList[j][1] += annList[i][j][1] / len(annList)

        reduced_annList.sort(key=lambda x: x[1], reverse=True) # decreasing sim
        target_supp_names = [item[0] for item in reduced_annList[:shot]]
        catPath = os.path.join(self.supp_root, str(catId))
        suppPaths = [os.path.join(catPath, supp_name+'.jpg') for supp_name in target_supp_names]
        suppImgs = [Image.open(suppPath).convert('RGB') for suppPath in suppPaths]
        if self.transform is not None:
            suppImgs = [self.transform(suppImg) for suppImg in suppImgs]

        return suppImgs

    def get_cats_in_img(self, imgId):
        annotationIds = self.coco.getAnnIds(imgIds=imgId, iscrowd=False)
        annotations = self.coco.loadAnns(annotationIds)
        catIds = [ann['category_id'] for ann in annotations if ann['category_id'] in self.json_cat_list]
        return list(set(catIds))
    
    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)
        cur_cat = self.chosen_cats[idx]
        cur_neg_cat = self.chosen_neg_cats[idx]
        img_id = self.ids[idx]

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0 and obj['area']>=32*32] # remove small objects
        # filter catId in annotation
        if self.is_train:
            anno = [obj for obj in anno if obj["category_id"] == cur_cat or obj["category_id"] == cur_neg_cat]
        else:
            anno = [obj for obj in anno if obj["category_id"] == cur_cat ]
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        classes = []
        for obj in anno:
            if obj["category_id"] == cur_cat:
                classes.append(1)
            else:
                classes.append(2)
        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        target = target.clip_to_image(remove_empty=True)

        if self.choose_close:
            img_supp = self.get_close_item_from_cat(img_id, cur_cat, shot=self.shot)
        else:
            img_supp = self.get_random_item_from_cat(cur_cat, excludeImgId=img_id, shot=self.shot)

        if self.neg_supp:
            img_neg_supp = self.get_random_item_from_cat(cur_neg_cat, excludeImgId=img_id, shot=self.shot)

        # save for visualization
        if self.save_img:
            img.save(self.dirName+'{:08d}query.jpg'.format(idx))
            for s in range(self.shot):
                img_supp[s].save(self.dirName+'{:08d}supp{:02d}.jpg'.format(idx, s))
                if self.neg_supp:
                    img_neg_supp[s].save(self.dirName+'{:08d}neg_supp{:02d}.jpg'.format(idx, s))

        if self._transforms is not None:
            img, target = self._transforms(img, target) 
            for s in range(self.shot):
                img_supp[s], _ = self._supp_transforms(img_supp[s], target) # dummy target
                if self.neg_supp:
                    img_neg_supp[s], _ = self._supp_transforms(img_neg_supp[s], target) # dummy target
            
        results = {}
        results['img'] = img
        results['img_supp'] = img_supp
        if self.neg_supp:
            results['img_neg_supp'] = img_neg_supp
        else:
            results['img_neg_supp'] = img_supp
        assert results['img_neg_supp']  is not None
        results['target'] = target
        # results['target_supp'] = target_supp
        results['idx'] = idx

        # ## debug
        # img_isnan = torch.sum(torch.isnan(img))
        # assert img_isnan == 0, img
        # for img_sup in img_supp:
        #     img_sup_isnan = torch.sum(torch.isnan(img_sup))
        #     assert img_sup_isnan == 0, img_sup
        # for img_sup in results['img_neg_supp']:
        #     img_sup_isnan = torch.sum(torch.isnan(img_sup))
        #     assert img_sup_isnan == 0, img_sup

        return results

    def get_img_info(self, index):
        # return query info
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        # print("index {} -> img id {}".format(index, img_id))
        img_cur_cat = self.chosen_cats[index]
        return img_data, img_cur_cat
