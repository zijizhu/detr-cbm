import os
import torch
import argparse
from tqdm import tqdm
import torchvision.ops as ops
from detr.models.detr import PostProcess
from detr.models.matcher import HungarianMatcher


def process_batch_outputs(matcher, postprocess, outputs, orig_size, scale, area_thresh, confidence, square):
    if orig_size:
        target_sizes = torch.stack([t['orig_size'] for t in outputs['targets']], dim=0)
    else:
        target_sizes = torch.stack([t['size'] for t in outputs['targets']], dim=0)

    batch_matched_idxs = matcher(outputs['outputs'], outputs['targets'])

    batch_results = postprocess(outputs['outputs'], target_sizes)
    batch_targets = outputs['targets']
    batch_features = outputs['h']

    batch_processed = []
    for matched_idxs, features, res, tgts, tgt_size in zip(batch_matched_idxs,
                                                           batch_features,
                                                           batch_results,
                                                           batch_targets,
                                                           target_sizes):
        pred_match_idxs, gt_match_idxs = matched_idxs

        pred_scores, pred_boxes, pred_labels = res['scores'], res['boxes'], res['labels']
        gt_boxes, gt_labels, gt_areas = tgts['boxes'], tgts['labels'], tgts['area']

        h, w = tgt_size
        img_id = tgts['image_id'].item()
        
        # Only keep matched features
        features = features[pred_match_idxs]
        pred_scores = pred_scores[pred_match_idxs]
        pred_boxes = pred_boxes[pred_match_idxs]
        pred_labels = pred_labels[pred_match_idxs]

        gt_labels, gt_areas = gt_labels[gt_match_idxs], gt_areas[gt_match_idxs]
        gt_boxes_cxcywh = gt_boxes[gt_match_idxs]
        
        # Process boxes
        crop_boxes_cxcywh_scaled = gt_boxes_cxcywh * torch.tensor([1, 1, scale, scale]) * torch.tensor([w, h, w, h])

        if square:
            box_wh = crop_boxes_cxcywh_scaled[:,2:4]
            box_wh_max, _ = torch.max(box_wh, dim=-1)
            crop_boxes_cxcywh_scaled[:,2:4] = box_wh_max.unsqueeze(1).repeat(1, 2)

        crop_boxes = (ops.box_convert(crop_boxes_cxcywh_scaled, 'cxcywh', 'xyxy')
                    .clamp(min=torch.tensor([0,0,0,0]), max=torch.tensor([w,h,w,h]))
                    .to(int))
        
        # Only keep objects with large enough area size in the image
        keep = (gt_areas > area_thresh) & (pred_scores > confidence)
        features = features[keep]
        crop_boxes = crop_boxes[keep]
        gt_labels = gt_labels[keep]
        gt_areas = gt_areas[keep]

        batch_processed.append({'detr_features': features,
                                'crop_boxes': crop_boxes,
                                'gt_labels': gt_labels,
                                'gt_areas': gt_areas,
                                'img_id': img_id,
                                'img_size': {'height': h, 'width': w}})
    return batch_processed


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script for building data for training detr-clip projector')

    parser.add_argument('--square', action='store_true', help="Make the crop boxes square")
    parser.add_argument('--orig_size', action='store_true', help="Use the original image size")
    parser.add_argument('--scale', default=1.2, type=float, help="Factor to scale bounding box sizes")
    parser.add_argument('--area_thresh', default=3000, type=int, help="Ignore small objects less than area threshold")
    parser.add_argument('--confidence_thresh', default=0.7, type=float, help="Ignore objects with low prediction confidence")

    # Same matcher arguments used in detr
    parser.add_argument('--cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    
    args = parser.parse_args()

    matcher = HungarianMatcher(cost_class=args.cost_class, cost_bbox=args.cost_bbox, cost_giou=args.cost_giou)
    postprocess = PostProcess()

    print('Processing training set...')
    for i in tqdm(range(1, 60)):
        detr_outputs = torch.load(os.path.join(args.input_dir, f'detr_outputs_part{i}.pth'))

        processed = []
        for batch in detr_outputs:
            processed += process_batch_outputs(matcher, postprocess, batch,
                                               args.orig_size, args.scale, args.area_thresh, args.confidence_thresh, args.square)
        torch.save(processed, os.path.join(args.output_dir, f'train_part{i}.pth'))

    # print('Processing validation set...')
    # detr_outputs = torch.load(os.path.join(args.input_dir, f'detr_outputs_val.pth'))
    # processed = process_batch_outputs(matcher, postprocess, detr_outputs, args.orig_size, args.scale, args.area_thresh)
    # torch.save(processed, os.path.join(args.output_dir, f'val.pth'))
