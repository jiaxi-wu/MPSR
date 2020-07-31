#!/bin/bash
# Warning: This scripts was used for hyperparamter tuning. 
# Only few-shot classes are limited to N-shot (not base) for more general sampling, may leading to a drop in perfromance.
# Hope this will help your for further research.
export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2
SEED=(0 1 2 3 4 5)
SPLIT=(1 2 3)
SHOT=(10 5 3 2 1)
for seed in ${SEED[*]} 
do
  mkdir fs_exp/voc_series_results_seed${seed}
  for shot in ${SHOT[*]} 
  do
    for split in ${SPLIT[*]} 
    do
      configfile=configs/fewshot/standard/e2e_voc_split${split}_${shot}shot_finetune.yaml
      python tools/fewshot_exp/datasets/voc_sample_series.py ${shot} ${split} ${seed}
      python tools/fewshot_exp/crops/create_crops_voc_standard.py ${shot}
      python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}
      rm model_final.pth
      rm last_checkpoint
      mv inference/voc_2007_test/result.txt fs_exp/voc_series_results_seed${seed}/result_split${split}_${shot}shot.txt
    done
  done
done
#python tools/fewshot_exp/cal_novel_voc.py
