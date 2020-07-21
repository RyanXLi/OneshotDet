# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        # transposed_batch = list(zip(*batch))
        images = [result['img'] for result in batch]
        # print('images', [image.size() for image in images])
        images = to_image_list(images, self.size_divisible)
        

        # images_support = [result['img_supp'] for result in batch]
        # images_support = to_image_list(images_support, self.size_divisible)

        # images_neg_support = [result['img_neg_supp'] for result in batch]
        # images_neg_support = to_image_list(images_neg_support, self.size_divisible)

        images_support = []
        images_neg_support = []
        for result in batch:
            images_support = images_support + result['img_supp']
            images_neg_support = images_neg_support + result['img_neg_supp']

        images_support = to_image_list(images_support, self.size_divisible)
        images_neg_support = to_image_list(images_neg_support, self.size_divisible)
        
        targets = [result['target'] for result in batch]
        # targets_support = [result['target_supp'] for result in batch]
        img_ids = [result['idx'] for result in batch]
        target_ids = [result['target_id'] for result in batch]

        return images, images_support, images_neg_support, targets, img_ids, target_ids
