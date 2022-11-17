# FocalNet for Object Detection with DINO

This repo contains the code for reproducing object detection results of our [FocalNets](https://arxiv.org/abs/2203.11926). It is based on [DINO](https://github.com/IDEA-Research/DINO).

## Installation

Please follow [DINO's instruction](https://github.com/IDEA-Research/DINO) for installation.

## Training

* Train on COCO with FocalNet-L with 3 focal levels:

```
python -m torch.distributed.launch --nproc_per_node={ngpus} main.py --config_file config/DINO/DINO_4scale_focalnet_fl3.py --coco_path {coco_path} --output_dir {output_dir}
```

* Train on COCO with FocalNet-L with 4 focal levels:

```
python -m torch.distributed.launch --nproc_per_node={ngpus} main.py --config_file config/DINO/DINO_4scale_focalnet_fl4.py --coco_path {coco_path} --output_dir {output_dir}
```

## Model Zoos

All models are provided in:

> **Focal Modulation Networks**: [Focal Modulation Networks Model Zoo](https://github.com/microsoft/FocalNet).

## Citation

If you find this repo useful to your project, please consider to cite it with following bib:

    @misc{yang2022focalnet,  
      author = {Yang, Jianwei and Li, Chunyuan and Dai, Xiyang and Yuan, Lu and Gao, Jianfeng},
      title = {Focal Modulation Networks},
      publisher = {arXiv},
      year = {2022},
    }

and also:

    @misc{zhang2022dino,
          title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection}, 
          author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
          year={2022},
          eprint={2203.03605},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }

