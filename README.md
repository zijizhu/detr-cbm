# DETR with Concept Bottleneck Models

## Setup

```bash
git clone https://github.com/zijizhu/detr.gits
git submodule update --init
bash download.sh
```

## Run Scripts

Build dataset:
```bash
python build_data_detr2clip.py --orig_size --square --input_dir detr_outputs --output_dir data
```

```
python main_inference.py --device cuda --output_dir outputs --resume checkpoints/detr-r50-e632da11.pth --coco_path coco
```
