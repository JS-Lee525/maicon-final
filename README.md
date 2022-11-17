# FocalNet for Object Detection with DINO

This repo contains the code for reproducing object detection results of FocalNets. It is based on [DINO](https://github.com/IDEA-Research/DINO).

## Installation

Please follow [DINO's instruction](https://github.com/IDEA-Research/DINO) for installation.

## Training

* Train on COCO with FocalNet-L:

```
python -m torch.distributed.launch --nproc_per_node={ngpus} main.py --config_file config/DINO/DINO_4scale_focalnet.py --coco_path {coco_path} --output_dir {output_dir}
```