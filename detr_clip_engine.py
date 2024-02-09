# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import detr.util.misc as utils
from detr.datasets.coco_eval import CocoEvaluator


@torch.no_grad()
def evaluate_and_save(model,
                      clip_model,
                      detr_criterion,
                      detr_postprocessors,
                      data_loader,
                      base_ds,
                      device,
                      output_dir,
                      split):
    model.eval()
    detr_criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    coco_evaluator = CocoEvaluator(base_ds, ('bbox',))

    num_processed = 0
    save_dicts = []
    for samples_detr, samples_clip, targets in metric_logger.log_every(data_loader, 10, header):
        samples_detr, samples_clip = samples_detr.to(device), samples_clip.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        clip_features = clip_model(samples_clip)

        hs, outputs = model(samples_detr)
    
        loss_dict = detr_criterion(outputs, targets)
        weight_dict = detr_criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = detr_postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        ##### Save outputs #####
        # Each iteration, outputs is a dict with tensors of [batch_size, ...] as values
        #                 targets is a list of dict, len(targets) == batch_size
        batch_size = hs.size(1)
        save_dicts.append({'detr_f': hs[-1].detach().cpu(),
                           'clip_f': clip_features.detach().cpu(),
                           'outputs': {k: v.detach().cpu() for k, v in outputs.items()},
                           'targets': [{k: v.detach().cpu() for k, v in t.items()} for t in targets]})
        
        num_processed += batch_size
        if num_processed == 5000:
            torch.save(save_dicts, os.path.join(output_dir, f'detr_clip_feats{num_processed // 5000}.pth'))
            del save_dicts
            save_dicts = []
        ##########

        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None and 'bbox' in detr_postprocessors.keys():
        stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    return stats, coco_evaluator