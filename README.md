# Multi-scale Positive Sample Refinement for Few-shot Object Detection, ECCV 2020

Our code is based on  [https://github.com/facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and developed with Python 3.6.5 & PyTorch 1.1.0.

## Abstract
Few-shot object detection (FSOD) helps detectors adapt to unseen classes with few training instances when manual annotation is time-consuming or data acquisition is limited.
In this work, we highlights the necessity of handling the problem of scale variations, which is challenging due to the unique sample distribution.
The lack of labels of novel classes leads to a sparse scale space which may be totally divergent from the original distribution of abundant training data. 
To this end, we propose a Multi-scale Positive Sample Refinement (MPSR) approach to enrich object scales in FSOD. 
It generates multi-scale positive samples as object pyramids and refines the prediction at various scales. For more details, please refer to our ECCV paper ([arxiv](https://arxiv.org/abs/2007.09384)). 


<div align=center>
<img src="https://github.com/jiaxi-wu/MPSR/blob/master/tools/fewshot_exp/MPSR_arch.jpg" width="600">
</div>

## Installation
Check INSTALL.md for installation instructions. Since maskrcnn-benchmark has been deprecated, please follow these instructions carefully (e.g. version of Python packages).

## Prepare datasets

### Prepare original Pascal VOC & MS COCO datasets
First, you need to download the VOC & COCO datasets.
We recommend to symlink the path of the datasets to `datasets/` as follows

We use `minival` and `valminusminival` sets from [Detectron](https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/data/README.md#coco-minival-annotations) ([filelink](https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz)).

```bash
# symlink the coco dataset
cd ~/github/maskrcnn-benchmark
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2014 datasets/coco/train2014
ln -s /path_to_coco_dataset/test2014 datasets/coco/test2014
ln -s /path_to_coco_dataset/val2014 datasets/coco/val2014

# for pascal voc dataset:
ln -s /path_to_VOCdevkit_dir datasets/voc
```

### Prepare base and few-shot datasets
For a fair comparison, we use the few-shot data splits from [Few-shot Object Detection via Feature Reweighting](https://github.com/bingykang/Fewshot_Detection) as a standard evaluation.
To download their data splits and transfer it into VOC/COCO style, you need to run this script:
```bash
bash tools/fewshot_exp/datasets/init_fs_dataset_standard.sh
```
This will also generate the datasets on base classes for base training.

## Training and Evaluation
4 scripts are used for full splits experiments and you can modify them later. 
They will crop objects and store them (e.g. `datasets/voc/VOC2007/Crops_standard`) before training.
You may need to change GPU device which is `export CUDA_VISIBLE_DEVICES=0,1` by default.
```bash
tools/fewshot_exp/
├── train_voc_base.sh
├── train_voc_standard.sh
├── train_coco_base.sh
└── train_coco_standard.sh
```

Configurations of base & few-shot experiments are:
```base
configs/fewshot/
├── base
│   ├── e2e_coco_base.yaml
│   └── e2e_voc_split*_base.yaml
└── standard
    ├── e2e_coco_*shot_finetune.yaml
    └── e2e_voc_split*_*shot_finetune.yaml
```
Modify them if needed. If you have any question about these parameters (e.g. batchsize), please refer to [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) for quick solutions.

### Perform few-shot training on VOC dataset
1. Run the following for base training on 3 VOC splits
```bash
bash tools/fewshot_exp/train_voc_base.sh
```
This will generate base models (e.g. `model_voc_split1_base.pth`) and corresponding pre-trained models (e.g. `voc0712_split1base_pretrained.pth`).

2. Run the following for few-shot fine-tuning
```bash
bash tools/fewshot_exp/train_voc_standard.sh
```
This will perform evaluation on 1/2/3/5/10 shot of 3 splits. 
Result folder is `fs_exp/voc_standard_results` by default, and you can get a quick summary by:
```bash
python tools/fewshot_exp/cal_novel_voc.py fs_exp/voc_standard_results
```

3. For more general experiments, refer to `tools/fewshot_exp/train_voc_series.sh`. In this script, only few-shot classes are limited to N-shot. This may lead to a drop in performance but more natural conditions.

### Perform few-shot training on COCO dataset
1. Run the following for base training
```bash
bash tools/fewshot_exp/train_coco_base.sh
```
This will generate the base model (`model_coco_base.pth`) and corresponding pre-trained model (`coco_base_pretrained.pth`).

2. Run the following for few-shot fine-tuning
```bash
bash tools/fewshot_exp/train_coco_standard.sh
```
This will perform evaluation on 10/30 shot. 
Result folder is `fs_exp/coco_standard_results` by default, and you can get a quick summary by:
```bash
python tools/fewshot_exp/cal_novel_coco.py fs_exp/coco_standard_results
```

**Notice:** Recently we find that [FSRW](https://github.com/bingykang/Fewshot_Detection) are not using the minival set for evaluation. So the training datasplits they provided contain few images in minival. Now we replace them while get almost the same evaluation results.

### Baseline experiments
For baseline experiments (i.e. Baseline-FPN in the paper), similar scripts are available as below:
```bash
tools/fewshot_exp/train_baseline_voc_base.sh
tools/fewshot_exp/train_baseline_voc_standard.sh
tools/fewshot_exp/train_baseline_coco_base.sh
tools/fewshot_exp/train_baseline_coco_standard.sh
```
Corresponding Cfgs are at `configs/fewshot_baseline/`, in which `MODEL.CLOSEUP_REFINE` is set to `False`. Result folder is `fs_exp/*_baseline_standard_results` by default.

### Pretrained weight files
Google Drive:
||Baseline-FPN|MPSR|
|:--:|:--:|:--:|
|VOC Split 1|[model_baseline_voc_split1_base.pth](https://drive.google.com/file/d/10wlsQw8AXoWALTbpQHfHEWzdl_TvyOIQ/view?usp=sharing)|[model_voc_split1_base.pth](https://drive.google.com/file/d/10CS_TbJC3KuUr-oftoS5-a-WLM-m9PQw/view?usp=sharing)|
|VOC Split 2|[model_baseline_voc_split2_base.pth](https://drive.google.com/file/d/1ziOWz65N5JB1pmB9q0Bz7Vr264oa60zN/view?usp=sharing)|[model_voc_split2_base.pth](https://drive.google.com/file/d/1lmDUmRs7OyaUPeOL5yaS80C-NgBnI6Fx/view?usp=sharing)|
|VOC Split 3|[model_baseline_voc_split3_base.pth](https://drive.google.com/file/d/1nytlH8m7xB3BVakRaVShmnXYt0gAtFzJ/view?usp=sharing)|[model_voc_split3_base.pth](https://drive.google.com/file/d/1juBJETAwoGEXI18PzlDdUXkOVzXZWeOF/view?usp=sharing)|

Or [BaiduYun](https://pan.baidu.com/s/12tqtB0DC-WGWw7ZB9F_jXw) with code: **vuef**.

## Citation
```
@inproceedings{wu2020mpsr,
  author = {Wu, Jiaxi and Liu, Songtao and Huang, Di and Wang, Yunhong},
  booktitle = {European Conference on Computer Vision},
  title = {Multi-Scale Positive Sample Refinement for Few-Shot Object Detection},
  year = {2020}
}
```
