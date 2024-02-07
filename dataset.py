import os
import torch
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


def collate_fn(batch):
    batch_detr_features = []
    batch_clip_imgs = []
    for detr_feat, img in batch:
        batch_detr_features.append(detr_feat)
        batch_clip_imgs.append(img)
    return torch.cat(batch_detr_features), torch.cat(batch_clip_imgs)