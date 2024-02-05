import sys
import math
import clip
import torch
from torch import nn
from typing import Iterable

import utils
from torch.utils.data import DataLoader
from dataset import Detr2ClipDataset, collate_fn


class CosSimCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity()
    
    def forward(self, outputs, targets):
        return torch.sum(self.cos(outputs, targets))


def train_one_epoch(model: torch.nn.Module, clip_encoder: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            targets = clip_encoder.encode_image(targets)

        outputs = model(samples)
        loss = criterion(outputs, targets, 1)

        # reduce losses over all GPUs for logging purposes

        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    lr, weight_decay = 1e-4, 1e-4
    clip_model, clip_preprocess = clip.load('RN50')
    model = nn.Linear(256, 1024)
    criterion = nn.CosineEmbeddingLoss()
    dataset = Detr2ClipDataset('data', 'coco', split='train', img_transforms=clip_preprocess)
    dataloader = DataLoader(dataset=dataset, batch_size=8, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    device = 'cpu'

    num_epochs = 100
    epoch_results = []
    for epoch in range(0, num_epochs):
        train_stats = train_one_epoch(model, clip_model, criterion, dataloader, optimizer, device, epoch)
        epoch_results.append((train_stats, model.state_dict()))
    
    best_loss, best_model = sorted(epoch_results, key=lambda x: x[0]['loss'])[0]
    torch.save(best_model, 'detr_r50_to_clip_r50_linear.pth')
