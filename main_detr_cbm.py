# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import clip
from torch.utils.data import DataLoader, DistributedSampler

from model import build_model
import detr_modified.util.misc as utils
from detr_modified.models.matcher import build_matcher
from dataset import DetrClipDataset, collate_fn
from engine_detr_cbm import evaluate, train_one_epoch
from detr_modified.models.detr import PostProcess, SetCriterion


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--clip_name', default='ViT-B/16', type=str, 
                        choices=['RN50', 'ViT-B/32', 'ViT-B/16'])
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    # * Transformer
    parser.add_argument('--num_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nhead', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    
    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                    help="Relative classification weight of the no-object class")

    return parser

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

    dataset_train = DetrClipDataset(dataset_dir=args.dataset_path, coco_dir=args.coco_path, split='train')
    dataset_val = DetrClipDataset(dataset_dir=args.dataset_path, coco_dir=args.coco_path, split='val')

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers)

    base_ds = dataset_val.coco

    # Build models
    clip_model, _ = clip.load(args.clip_name, device=device)
    concepts = []
    with open('data/concepts_cleaned_list.json', 'r') as fp:
        concepts = json.load(fp=fp)
    concepts_tokenized = clip.tokenize(concepts).to(device)
    concepts_encoded = clip_model.encode_text(concepts_tokenized)

    labels = ['a ' + base_ds.cats[i]['name'] if i in base_ds.cats else 'unknown' for i in range(91)] + ['unknown']
    labels_tokenized = clip.tokenize(labels).to(device)
    labels_encoded = clip_model.encode_text(labels_tokenized)

    d_clip = 1024 if args.clip_name == 'RN50' else 512
    model = build_model(256, d_clip, concepts_encoded.float(), args)

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    losses = ['labels', 'boxes', 'cardinality']

    matcher = build_matcher(args)
    criterion = SetCriterion(91, matcher, weight_dict, args.eos_coef, losses=losses)
    postprocessors = PostProcess()

    model.to(device)
    criterion.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, 
                                              data_loader_val, base_ds, device)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, output_dir / f'checkpoint_epoch{epoch}.pth')

        test_stats, coco_evaluator, val_outputs = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device
        )

        # Detach linear layer and perform detection using label embeddings directly
        evaluate(model,
                 criterion,
                 postprocessors,
                 data_loader_val,
                 base_ds,
                 device,
                 labels_encoded.float())
        if args.output_dir:
            val_output_paths = output_dir / f'val_outputs_epoch{epoch}.pth'
            utils.save_on_master(val_outputs, val_output_paths)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
