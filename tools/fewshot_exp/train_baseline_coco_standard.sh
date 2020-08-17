#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2
SHOT=(10 30)
mkdir fs_exp/coco_baseline_standard_results
for shot in ${SHOT[*]} 
do
  configfile=configs/fewshot_baseline/standard/e2e_coco_${shot}shot_finetune.yaml
  python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}
  rm model_final.pth
  rm last_checkpoint
  mv ~/coco_result.txt fs_exp/coco_baseline_standard_results/result_${shot}shot.txt
done
python tools/fewshot_exp/cal_novel_coco.py fs_exp/coco_baseline_standard_results
