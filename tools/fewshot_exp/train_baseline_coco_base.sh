#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2
configfile=configs/fewshot_baseline/base/e2e_coco_base.yaml
python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}
mv model_final.pth model_baseline_coco_base.pth
mv ~/coco_result.txt fs_exp/result_coco_base.txt
rm last_checkpoint
python tools/fewshot_exp/trans_baseline_coco_pretrained.py
