set -x

git submodule update --init
apt-get update -y && apt-get upgrade -y && apt-get install unzip zip -y
pip install scipy pycocotools ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git

cd data
gdown "1k8yrcd1qXNbkihtIJeCeuEPJ_i8fZeJq"
gdown "16-EtigRUMYasCZGv4QILnvTdbEi1fGp6"

unzip -qq detr_rn50_clip_vit-b16.zip
mv outputs/* .
rm detr_rn50_clip_vit-b16.zip
rmdir outputs
rmdir detr_rn50_clip_vit-b16

cd ../coco
wget -q --show-progress "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
unzip -qq annotations_trainval2017.zip
