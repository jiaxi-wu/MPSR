#!/bin/bash
#data split from code imp for "Few-shot Object Detection via Feature Reweighting, ICCV 2019"
git clone https://github.com/bingykang/Fewshot_Detection.git ../Fewshot_Detection
#init base/novel sets for fewshot exps
python tools/fewshot_exp/datasets/voc_create_base.py
python tools/fewshot_exp/datasets/voc_create_standard.py
python tools/fewshot_exp/datasets/coco_create_base.py
python tools/fewshot_exp/datasets/coco_create_standard.py
mkdir fs_exp
