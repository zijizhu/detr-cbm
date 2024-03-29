set -x

git submodule update --init
apt-get update -y && apt-get upgrade -y && apt-get install unzip zip -y
pip install scipy pycocotools ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git

cd coco
wget -q --show-progress "http://images.cocodataset.org/zips/train2017.zip"
wget -q --show-progress "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

unzip -qq train2017.zip
rm train2017.zip
unzip -qq annotations_trainval2017.zip
rm annotations_trainval2017.zip

cd ../checkpoints
wget -q --show-progress "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"