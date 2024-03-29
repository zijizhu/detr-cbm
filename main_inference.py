# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import clip
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import torchvision
import detr_modified.util.misc as utils
from detr_modified.datasets.coco import make_coco_transforms, ConvertCocoPolysToMask
from detr_modified.datasets import build_dataset, get_coco_api_from_dataset
from detr_clip_engine import evaluate_and_save
from detr_modified.models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--clip_name', default='RN50', type=str, 
                        choices=['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'])


    return parser


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, detr_transforms, clip_transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._detr_transforms = detr_transforms
        self._clip_transforms = clip_transforms
        self._detr_prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        detr_img, target = self._detr_prepare(img, target)
        if self._detr_transforms is not None:
            detr_img, target = self._detr_transforms(detr_img, target)
        
        if self._clip_transforms is not None:
            clip_img = self._clip_transforms(img)
        return detr_img, clip_img, target


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = utils.nested_tensor_from_tensor_list(batch[0])
    batch[1] = torch.stack(batch[1])
    return tuple(batch)


def build_dataset(image_set, clip_transforms, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json')
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder,
                            ann_file,
                            detr_transforms=make_coco_transforms(image_set, augmentation=args.augmentation),
                            clip_transforms=clip_transforms,
                            return_masks=args.masks)
    return dataset


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    clip_model, clip_preprocess = clip.load(args.clip_name, device=args.device)

    detr_model, criterion, postprocessors = build_model(args)
    detr_model.to(device)

    dataset_train = build_dataset(image_set='train', clip_transforms=clip_preprocess, args=args)
    dataset_val = build_dataset(image_set='val', clip_transforms=clip_preprocess, args=args)
    
    data_loader_train = DataLoader(dataset_train, args.batch_size, drop_last=False,
                                   collate_fn=collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, drop_last=False,
                                 collate_fn=collate_fn, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        detr_model.load_state_dict(checkpoint['model'])

    if args.split == 'train':
        base_ds = get_coco_api_from_dataset(dataset_train)
        coco_evaluator = evaluate_and_save(detr_model, clip_model, criterion, postprocessors,
                                           data_loader_train, base_ds, device, args.output_dir, 'train')
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)
        coco_evaluator = evaluate_and_save(detr_model, clip_model, criterion, postprocessors,
                                           data_loader_val, base_ds, device, args.output_dir, 'val')
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
