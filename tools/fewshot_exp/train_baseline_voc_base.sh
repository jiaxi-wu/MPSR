#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2
SPLIT=(1 2 3)
for split in ${SPLIT[*]} 
do
  configfile=configs/fewshot_baseline/base/e2e_voc_split${split}_base.yaml
  python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}
  mv model_final.pth model_baseline_voc_split${split}_base.pth
  mv inference/voc_2007_test_split${split}_base/result.txt fs_exp/result_split${split}_base.txt
  rm last_checkpoint
  python tools/fewshot_exp/trans_baseline_voc_pretrained.py ${split}
done
