# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from tqdm import tqdm
from typing import Iterable

import torch

import detr.util.misc as utils
from detr.datasets.coco_eval import CocoEvaluator


@torch.no_grad()
def evaluate_and_save(detr_model,
                      clip_model,
                      detr_criterion,
                      detr_postprocessors,
                      data_loader,
                      base_ds,
                      device,
                      output_dir,
                      split):
    detr_model.eval()
    detr_criterion.eval()

    coco_evaluator = CocoEvaluator(base_ds, ('bbox',))

    num_processed = 0
    save_dicts = []
    for samples_detr, samples_clip, targets in tqdm(data_loader):
        samples_detr, samples_clip = samples_detr.to(device), samples_clip.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        clip_features = clip_model.encode_image(samples_clip)

        hs, outputs = detr_model(samples_detr)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = detr_postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        ##### Save outputs #####
        # Each iteration, outputs is a dict with tensors of [batch_size, ...] as values
        #                 targets is a list of dict, len(targets) == batch_size
        outputs = [{'pred_logits': logits, 'pred_boxes': boxes}
                   for logits, boxes
                   in zip(outputs['pred_logits'].detach().cpu(), outputs['pred_boxes'].detach().cpu())]
        
        batch_size = hs.size(1)
        assert (len(clip_features) == batch_size and
                len(outputs) == batch_size and
                len(targets) == batch_size)
        
        for detr_f, clip_f, out, tgt, in zip(hs[-1].detach().cpu(),
                                             clip_features.detach().cpu(),
                                             outputs,
                                             targets):
            save_dicts.append({'detr_f': detr_f,
                               'clip_f': clip_f,
                               'outputs': out,
                               'targets': {k: v.detach().cpu() for k, v in tgt.items()}})
        
        num_processed += batch_size
        if num_processed % 5000 == 0:
            torch.save(save_dicts, os.path.join(output_dir, f'detr_clip_{split}{num_processed // 5000}.pth'))
            del save_dicts
            save_dicts = []
        ##########

        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    return coco_evaluator