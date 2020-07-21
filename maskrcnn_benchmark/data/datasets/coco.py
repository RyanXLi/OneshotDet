# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from maskrcnn_benchmark.utils.comm import get_rank

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

import random
import os
import pickle
import time
import numpy as np
import pickle
import glob

import logging
from PIL import Image, ImageEnhance, ImageOps, ImageFile

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


class COCODataset(torchvision.datasets.coco.CocoDetection):
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
        super(COCODataset, self).__init__(root, ann_file)
        self.rank = get_rank()
        random.seed(6666)
        remove_images_without_annotations = True

        self.cfg                    = cfg
        self.neg_supp               = cfg.FEW_SHOT.NEG_SUPPORT.TURN_ON
        self.neg_supp_num_cls       = cfg.FEW_SHOT.NEG_SUPPORT.NUM_CLS
        self.choose_close           = cfg.FEW_SHOT.CHOOSE_CLOSE
        self.choose_selected        = cfg.FEW_SHOT.CHOOSE_SELECTED
        self.shot                   = cfg.FEW_SHOT.NUM_SHOT
        if cfg.FEW_SHOT.SUPP_AUG:
            self.actual_num_imgs    = self.shot * (1 + cfg.FEW_SHOT.NUM_SUPP_AUG)
        else:
            self.actual_num_imgs    = self.shot
        self.save_img               = cfg.FEW_SHOT.SAVE_IMAGE
        self.training_exclude_cats  = cfg.FEW_SHOT.TRAINING_EXCL_CATS
        self.test_exclude_cats      = cfg.FEW_SHOT.TEST_EXCL_CATS
        self.supp_aug               = cfg.FEW_SHOT.SUPP_AUG
        self.is_train               = is_train
        self.selected_cls           = cfg.FEW_SHOT.TEST_SELECTED_CLS # from 1
        self.selected_order         = cfg.FEW_SHOT.TEST_SELECTED_SUPP

        if self.rank == 0:
            print('Save image: '+str(self.save_img))
            print('number of shots: '+str(self.shot))
            print('Use negative supp: '+str(self.neg_supp))
            print('Choose close supp: '+str(self.choose_close))

        logger = logging.getLogger("maskrcnn_benchmark.inference")

        # train-test split
        # voc_class = [1, 2, 3, 4, 5, 6, 7, 9, 40, 57, 59, 61, 63, 15, 16, 17, 18, 19, 20] # in terms of 1-80
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
        print('coco id', self.coco.getCatIds())
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

        task1_test_split_file = "task1_test_split.txt"
        with open(task1_test_split_file, 'r') as f:
            task1_test_imgs = [line.split(' ')[0] for line in f]


        self.json_cat_list = list(self.json_category_id_to_contiguous_id.keys()) # 1-80
        self.catalog = {} # cat to list of img_ids
        for cat in self.json_cat_list:
            self.catalog[cat] = []
            img_ids_cur_cat = self.coco.getImgIds(catIds=cat)
            if self.cfg.FEW_SHOT.TASK==1 and not is_train:
                # path = self.coco.loadImgs(img_id)[0]['file_name']
                img_ids_cur_cat = [img_id for img_id in img_ids_cur_cat if \
                                        self.coco.loadImgs(img_id)[0]['file_name'] in task1_test_imgs]
            img_ids_cur_cat = sorted(img_ids_cur_cat)
            # filter images without detection annotations
            if remove_images_without_annotations:
                for img_id in img_ids_cur_cat:
                    ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat, iscrowd=False) # filter out iscrowd
                    anno = self.coco.loadAnns(ann_ids)
                    if has_valid_annotation(anno):
                        self.catalog[cat].append(img_id)
                        # constrain each cat to only have at most 2000 images
                        # prevent overfitting on cats having more images
                        # if len(self.catalog[cat]) >= 2000:
                        #     break

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
            if self.selected_cls != -1 and cat != self.selected_cls:
                continue
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
   
        # debug_check = False
        # if debug_check:
        #     for i in range(len(self.ids)):
        #         img_id = self.id_to_img_map[i]
        #         cur_cat = self.chosen_cats[i]

        #         ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cur_cat, iscrowd=False) # filter out iscrowd
        #         anno = self.coco.loadAnns(ann_ids)
        #         if not anno:
        #             print('ERROR: no gt')
        #             print(i)
        #             print(img_id)
        #             print(cur_cat)
        #         loaded_img = self.coco.loadImgs([img_id])
        #         if not loaded_img:
        #             print('no_img')
        #             print(i)
        #             print(img_id)
        #             print(cur_cat)
        #         # path = loaded_img[0]['file_name']
        
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
            self.close_dict = {}
            for cat in self.json_cat_list:
                if self.is_train:
                    close_dict_path = '/data/linz/few_shot/fcos_plus/fcos_plus/supp_sim/supp_similarity_'+str(cat)+'.pkl'
                else:
                    close_dict_path = '/data/linz/few_shot/fcos_plus/fcos_plus/supp_sim_test/supp_similarity_'+str(cat)+'.pkl'
                with open(close_dict_path, 'rb') as f:
                    self.close_dict[cat] = pickle.load(f)
            print('loaded all similarity')
            self.supp_root = '/data/linz/few_shot/fcos_plus/fcos_plus/supps'  # for cpu2
            # self.supp_root = '/data/xlide/fcos/supps' # for cpu1

        cats = self.coco.loadCats(self.json_cat_list)
        names=[cat['name'] for cat in cats]

        # get selected test image id
        if self.choose_selected:
            select_supp_test_dir = 'supps_test_selected'
            selected_supps = []
            for i in range(1, 21):
                cls_imgs = glob.glob(select_supp_test_dir+'/'+str(i)+'/*.jpg')
                cls_imgs.sort()
                selected_supps.append(cls_imgs)
            self.selected_supps = selected_supps
            self.selected_supp_ids = [
                        17, 13, 29, 2, 3, 
                        1, 7, 15, 4, 19,
                        5, 17, 15, 5, 6,
                        0, 7, 12, 16, 9]

        self.supp_aug_transforms = []
        if self.supp_aug:
            self.supp_aug_transforms.append(lambda x: x.transpose(Image.FLIP_LEFT_RIGHT)) # horizontal flip
            if self.cfg.FEW_SHOT.NUM_SUPP_AUG > 1:
                assert self.cfg.FEW_SHOT.NUM_SUPP_AUG==3
                self.supp_aug_transforms.append(self.color_jitter) # color operation
            if cfg.FEW_SHOT.NUM_SUPP_AUG==1:
                assert(len(self.supp_aug_transforms) == cfg.FEW_SHOT.NUM_SUPP_AUG)
            elif cfg.FEW_SHOT.NUM_SUPP_AUG==3:
                assert(len(self.supp_aug_transforms)+1 == cfg.FEW_SHOT.NUM_SUPP_AUG)

    def color_jitter(self, image):
        random_factor = np.random.uniform(0.1, 2)
        color_image = ImageEnhance.Color(image).enhance(random_factor)
        random_factor = np.random.uniform(0.1, 2)
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.uniform(0.1, 2)
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        random_factor = np.random.uniform(0.1, 2)
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)

    def get_random_item_from_cat(self, catId, excludeImgId, shot=1, idx=None):
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

        if self.cfg.FEW_SHOT.MASK_SUPP:
            for i, suppAnn in enumerate(valid_anns):
                print(suppAnn)
                mask = self.coco.annToMask(suppAnn)
                mask += self.coco.annToMask(suppAnn)
                mask = Image.fromarray(mask * 255, mode='L').convert('1')
                mask.save(self.dirName+'{:08d}supp_mask.jpg'.format(idx))
                imgs[i] = np.array(imgs[i])
                mask = np.array(mask)[:,:,None]
                imgs[i] = imgs[i] * mask
                imgs[i] = Image.fromarray(imgs[i])

        imgs = [imgs[i].crop((coords[i][0], coords[i][1], coords[i][0]+coords[i][2], coords[i][1]+coords[i][3])) for i in range(shot)]

        if self.supp_aug:
            augmented_suppImgs = []
            for supp in imgs:
                augmented_suppImgs.append(supp)
                for T in self.supp_aug_transforms:
                    augmented_suppImgs.append(T(supp))
            imgs = augmented_suppImgs

        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]

        return imgs

    def get_selected_item_from_cat(self, catId, shot=1):
        candidate_supps = self.selected_supps[catId-1]
        select_order = self.selected_supp_ids[catId-1]
        selected_supp = candidate_supps[select_order]
        imgs = [Image.open(selected_supp).convert('RGB')]

        if self.cfg.FEW_SHOT.MASK_SUPP:
            mask_file = os.path.join('voc2007_test_coco/Save', selected_supp.split('/')[-1])
            mask_img = Image.open(mask_file).convert('RGB')
            imgs = [mask_img]

        if self.supp_aug:
            augmented_suppImgs = []
            for supp in imgs:
                augmented_suppImgs.append(supp)
                for T in self.supp_aug_transforms:
                    augmented_suppImgs.append(T(supp))
            imgs = augmented_suppImgs

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

        if self.cfg.FEW_SHOT.MASK_SUPP:
            suppAnnIds = [int(supp_name.split('_')[1].split('.')[0]) for supp_name in target_supp_names]
            suppAnns = [self.coco.loadAnns(suppAnnId) for suppAnnId in suppAnnIds]
            for i, suppAnn in enumerate(suppAnns):
                mask = self.coco.annToMask(suppAnn[0])
                assert len(suppAnn) == 1, len(suppAnn)
                bbox = suppAnn[0]['bbox']
                for j in range(len(suppAnn)):
                    mask += self.coco.annToMask(suppAnn[j])
                mask = Image.fromarray(mask * 255, mode='L').convert('1')
                mask = mask.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))

                suppImgs[i] = np.array(suppImgs[i])
                mask = np.array(mask)[:,:,None]
                suppImgs[i] *= mask
                suppImgs[i] = Image.fromarray(suppImgs[i])

        if self.supp_aug:
            augmented_suppImgs = []
            for supp in suppImgs:
                augmented_suppImgs.append(supp)
                # for T in self.supp_aug_transforms:
                if self.cfg.FEW_SHOT.NUM_SUPP_AUG == 1:
                    augmented_suppImgs.append(self.supp_aug_transforms[0](supp)) # flip
                elif self.cfg.FEW_SHOT.NUM_SUPP_AUG == 3:
                    augmented_suppImgs.append(self.supp_aug_transforms[0](supp)) # flip
                    augmented_suppImgs.append(self.supp_aug_transforms[1](supp)) # color jitering of non-flip
                    augmented_suppImgs.append(self.supp_aug_transforms[1](augmented_suppImgs[1])) # color jitering of flip
                else:
                    raise Exception('NUM_SUPP_AUG invalid')
            suppImgs = augmented_suppImgs

        if self.transform is not None:
            suppImgs = [self.transform(suppImg) for suppImg in suppImgs]

        # print('len_supp', len(suppImgs))
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
        anno = [obj for obj in anno if obj["iscrowd"] == 0 ] # remove small objects and obj['area']>=32*32
        # filter catId in annotation
        if self.is_train:
            anno = [obj for obj in anno if obj["category_id"] == cur_cat]  # change new version of neg support linz
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
        elif self.choose_selected:
            img_supp = self.get_selected_item_from_cat(cur_cat, shot=self.shot)
        else:
            img_supp = self.get_random_item_from_cat(cur_cat, excludeImgId=img_id, shot=self.shot, idx=idx)

        if self.neg_supp:
            img_neg_supp = self.get_random_item_from_cat(cur_neg_cat, excludeImgId=img_id, shot=self.shot)

        # save for visualization
        if self.save_img:
            img.save(self.dirName+'{:08d}query.jpg'.format(idx))
            for s in range(self.actual_num_imgs):
                img_supp[s].save(self.dirName+'{:08d}supp{:02d}.jpg'.format(idx, s))
                if self.neg_supp:
                    img_neg_supp[s].save(self.dirName+'{:08d}neg_supp{:02d}.jpg'.format(idx, s))

        if self._transforms is not None:
            img, target = self._transforms(img, target) 
            for s in range(self.actual_num_imgs):
                img_supp[s], _ = self._supp_transforms(img_supp[s], target) # dummy target
                if self.neg_supp:
                    img_neg_supp[s], _ = self._supp_transforms(img_neg_supp[s], target) # dummy target
            
        results = {}
        results['img'] = img
        results['img_supp'] = img_supp
        if self.neg_supp and self.is_train:
            results['img_neg_supp'] = img_neg_supp
        else:
            results['img_neg_supp'] = img_supp
        assert results['img_neg_supp']  is not None
        results['target'] = target
        # results['target_supp'] = target_supp
        results['idx'] = idx

        results['target_id'] = cur_cat

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

    # def __len__(self):
    #     return 400

    def get_img_info(self, index):
        # return query info
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        # print("index {} -> img id {}".format(index, img_id))
        img_cur_cat = self.chosen_cats[index]
        return img_data, img_cur_cat
