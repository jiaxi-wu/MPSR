#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2
configfile=configs/fewshot/base/e2e_coco_base.yaml
python tools/fewshot_exp/crops/create_crops_coco_base.py
python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}
mv model_final.pth model_coco_base.pth
mv ~/coco_result.txt fs_exp/result_coco_base.txt
rm last_checkpoint
python tools/fewshot_exp/trans_coco_pretrained.py
