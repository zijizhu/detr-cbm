import clip
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from pycocotools.coco import COCO

from cocoeval import CocoEvaluator
from detr.models.detr import PostProcess
from detr.models.matcher import HungarianMatcher

clip_model, clip_preprocess = clip.load('RN50', device='cpu')
coco_val = COCO(annotation_file='coco/annotations/instances_val2017.json')

model = nn.Linear(256, 1024)
model.load_state_dict(torch.load('checkpoints/detr_r50_to_clip_r50_linear_epoch4.pth')['model'])

if __name__ == '__main__':
    val = torch.load('detr/outputs/detr_outputs_val.pth')


    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    postprocess = PostProcess()

    texts = ['a ' + coco_val.cats[i]['name'] if i in coco_val.cats else 'unknown' for i in range(91)] + ['unknown']
    idx2cocoid = [k for k in coco_val.cats]
    texts_tokenized = clip.tokenize(texts)
    texts_encoded = clip_model.encode_text(texts_tokenized)

    # Original detr performance
    coco_evaluator = CocoEvaluator(coco_val, ('bbox',))
    for batch in tqdm(val):
        target_sizes = torch.stack([t['orig_size'] for t in batch['targets']], dim=0)
        batch_matched_idxs = matcher(batch['outputs'], batch['targets'])
        batch_results = postprocess(batch['outputs'], target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(batch['targets'], batch_results)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # detr2clip performance
    coco_evaluator = CocoEvaluator(coco_val, ('bbox',))
    for batch in tqdm(val):
        target_sizes = torch.stack([t['orig_size'] for t in batch['targets']], dim=0)
        batch_matched_idxs = matcher(batch['outputs'], batch['targets'])
        batch_results = postprocess(batch['outputs'], target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(batch['targets'], batch_results)}

        for i, results in enumerate(batch_results):
            features, img_id = batch['h'][i], batch['targets'][i]['image_id'].item()

            with torch.no_grad():
                logits = model(features) @ texts_encoded.T
                clip_probs = F.softmax(logits, dim=-1)
                values, labels = clip_probs.max(dim=-1)
            
            res[img_id]['labels'] = labels

        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()