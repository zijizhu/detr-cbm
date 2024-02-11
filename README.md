# DETR with Concept Bottleneck Models

## Setup

```bash
git clone https://github.com/zijizhu/detr.gits
git submodule update --init
bash download.sh
```

## Run Scripts

Run detr inference on coco train and val
```bash
python main_inference.py --device cuda --output_dir outputs --resume checkpoints/detr-r50-e632da11.pth --coco_path coco
```

Run training script
```bash
python main_inference.py --device cuda --output_dir outputs --resume checkpoints/detr-r50-e632da11.pth --coco_path coco
```
