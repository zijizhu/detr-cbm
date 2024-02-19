# DETR with Concept Bottleneck Models

## Setup

```bash
git clone https://github.com/zijizhu/detr-cbm.git
cd detr-cbm
git submodule update --init
bash download_train.sh
```

## Run Scripts

Run detr inference on coco train and val
```bash
python main_inference.py --device cuda --output_dir outputs --resume checkpoints/detr-r50-e632da11.pth --coco_path coco
```

Run training script with detr's default hyperparameters
```bash
python main_detr_cbm.py --concepts --dataset_path data --coco_path coco --output_dir outputs --device cpu
```

Run training script with clip4hoi's default hyperparameters
```bash
python main_detr_cbm.py --dataset_path data --coco_path coco --output_dir outputs --device cpu
```
