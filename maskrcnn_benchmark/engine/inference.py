# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import numpy as np
import cv2
from PIL import Image

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str


def overlay_boxes(image, boxes, scores=None, color=(255, 225, 225)):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """


    if scores is not None:
        assert len(boxes) == len(scores)
    for ind, box in enumerate(boxes):
        box = box.to(torch.int32)
        
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image.copy(), tuple(top_left), tuple(bottom_right), color, 2
        )
        if scores is not None:
            score = scores[ind]
            cv2.putText(image, '%.3f' % (score), (top_left[0], top_left[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

    return image

def compute_on_dataset(model, data_loader, device, timer=None, output_folder=None, stop_iter=0):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")

    for counter, (images, images_support, images_neg_support, targets, image_ids, target_ids) in enumerate(tqdm(data_loader)):
        images = images.to(device)
        images_support = images_support.to(device) #[item.to(device) for item in images_support]
        images_neg_support = images_neg_support.to(device)#[item.to(device) for item in images_neg_support]

        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            if timer:
                timer.tic()
            output = model(images, images_support, targets, images_neg_support, device=device, target_ids=target_ids)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )


        #######################
        ##### Visualization
        #######################
        # images = images.to(cpu_device)
        # image = images.tensors[0].contiguous().numpy()
        # mean_img = np.asarray([102.9801, 115.9465, 122.7717])[:, None, None] 
        # image += mean_img
        # image = np.rollaxis(image, 2)
        # image = np.rollaxis(image, 2)
        # image = image.astype('uint8')
        # cv2.imwrite("{}/query".format(output_folder)+str(counter)+".jpg", image)
        # # print(image)
        # images_support = images_support.tensors[0].to(cpu_device)
        # image_support = images_support.numpy()
        # image_support += mean_img
        # image_support = np.rollaxis(image_support, 2)
        # image_support = np.rollaxis(image_support, 2)
        # image_support = image_support.astype('uint8')
        # cv2.imwrite("{}/supp".format(output_folder)+str(counter)+".jpg", image_support)
        # targets = [target.to(cpu_device) for target in targets]
        # # print('target:')
        # # print(targets[0].bbox, targets[0].get_field('labels'))
        # # print('output:')
        # # print(output[0].get_field("scores"))
        # # print()
        # # image = overlay_boxes(image, output[0].bbox)
        # image = overlay_boxes(image, targets[0].bbox, color=(0,0,255)) # r
        # labels = output[0].get_field("labels").tolist()
        # tp_boxes = []
        # fp_boxes = []
        # other_boxes = []
        # tp_scores = []
        # fp_scores = []
        # other_scores = []
        # scores = output[0].get_field("scores").tolist()
        # # max_score = max(scores)
        # # norm_scores = [score / max_score for score in scores]
        # chosen_idx = np.argsort(-np.asarray(scores))[:10]

        # for i in chosen_idx:
        #     box = output[0].bbox[i]
        #     score = output[0].get_field("scores")[i].item()
        #     if labels[i] == 1:
        #         tp_boxes.append(box)
        #         tp_scores.append(score)
        #     elif labels[i] == 0:
        #         fp_boxes.append(box)
        #         fp_scores.append(score)
        #     else:
        #         other_boxes.append(box)
        #         other_scores.append(score)
        # image = overlay_boxes(image, tp_boxes, scores=tp_scores, color=(255,0,0)) # b
        # image = overlay_boxes(image, fp_boxes, scores=fp_scores, color=(0,255,255)) # ye
        # image = overlay_boxes(image, other_boxes, scores=other_scores, color=(0,255,0)) # g
        # cv2.imwrite("{}/result".format(output_folder)+str(counter)+".jpg", image)
        
        # stop_iter = 100
        if stop_iter and counter >= stop_iter:
            return results_dict


    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        stop_iter=0, # if 0 finish whole dataset
        meters=None 
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer, output_folder=output_folder, stop_iter=stop_iter)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
