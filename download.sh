set -x

git submodule update --init
cd coco
apt-get update -y && apt-get upgrade -y && apt-get install unzip zip -y
wget -q --show-progress "http://images.cocodataset.org/zips/train2017.zip"
wget -q --show-progress "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

unzip -qq train2017.zip
rm train2017.zip
unzip -qq annotations_trainval2017.zip
rm annotations_trainval2017.zip

pip install scipy pycocotools ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
