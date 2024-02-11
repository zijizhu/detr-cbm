import os
import torch
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class Detr2ClipDataset(Dataset):
    def __init__(self, data_dir, coco_dir, split='train', img_transforms=None, return_pil=False, return_info=False) -> None:
        super().__init__()
        self.train_data = []
        self.return_pil = return_pil
        self.return_info = return_info
        if split == 'train':
            coco_ann_dir = os.path.join(coco_dir, 'annotations', 'instances_train2017.json')
            for i in range(1, 60):
                train_part_i = torch.load(os.path.join(data_dir, f'train_part{i}.pth'))
                self.train_data += train_part_i
            self.img_dir = os.path.join(coco_dir, 'train2017')
        elif split == 'val':
            coco_ann_dir = os.path.join(coco_dir, 'annotations', 'instances_val2017.json')
            self.train_data += torch.load(os.path.join(data_dir, f'val.pth'))
            self.img_dir = os.path.join(coco_dir, 'val2017')
        else:
            raise NotImplementedError
        
        # Filter out samples with no data (images with no large enough objects)
        self.train_data = [sample for sample in self.train_data if sample['detr_features'].size(0) > 0]

        self.coco = COCO(coco_ann_dir)

        self.img_transforms = img_transforms
    
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        sample = self.train_data[idx]
        img_id =sample['img_id']
        img_fn = self.coco.imgs[img_id]['file_name']
        img = Image.open(os.path.join(self.img_dir, img_fn)).convert('RGB')
        all_cropped_imgs = []
        for box in sample['crop_boxes'].tolist():
            cropped = img.crop(box)
            cropped_transformed = self.img_transforms(cropped)
            all_cropped_imgs.append(cropped_transformed)
        if self.return_info:
            return (sample['detr_features'],
                    all_cropped_imgs if self.return_pil else torch.stack(all_cropped_imgs),
                    img_id,
                    sample['gt_labels'])
        return sample['detr_features'], all_cropped_imgs if self.return_pil else torch.stack(all_cropped_imgs)


def collate_fn_detr2clip(batch):
    batch_detr_features = []
    batch_clip_imgs = []
    for detr_feat, img in batch:
        batch_detr_features.append(detr_feat)
        batch_clip_imgs.append(img)
    return torch.cat(batch_detr_features), torch.cat(batch_clip_imgs)


class DetrClipDataset(Dataset):
    def __init__(self, dataset_dir, coco_dir, split) -> None:
        self.data = []
        assert split in ['train', 'val']
        ann_filename = 'instances_train2017.json' if split == 'train' else 'instances_val2017.json'
        self.coco = COCO(os.path.join(coco_dir, 'annotations', ann_filename))
        print('Loading preprocessed samples into memory...')
        sample_filenames = [fn for fn in os.listdir(dataset_dir) if f'detr_clip_{split}' in fn]
        for fn in tqdm(sample_filenames):
            self.data += torch.load(os.path.join(dataset_dir, fn))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        detr_f, clip_f = item['detr_f'], item['clip_f']
        outputs, targets = item['outputs'],item['targets']
        detr_logits, detr_boxes = outputs['pred_logits'], outputs['pred_boxes']
        return detr_f, clip_f.float(), detr_logits, detr_boxes, targets

def collate_fn(batch):
    batch = list(zip(*batch))
    batch = [torch.stack(item) for item in batch[:-1]] + [batch[-1]]
    return batch