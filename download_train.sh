set -x

git submodule update --init
apt-get update -y && apt-get upgrade -y && apt-get install unzip zip -y
pip install scipy pycocotools ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git

cd data
gdown "1k8yrcd1qXNbkihtIJeCeuEPJ_i8fZeJq"
gdown "16-EtigRUMYasCZGv4QILnvTdbEi1fGp6"

unzip -qq detr_rn50_clip_vit-b16.zip
mv detr_rn50_clip_vit-b16/* .
rm detr_rn50_clip_vit-b16.zip
