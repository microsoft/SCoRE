#! /bin/bash

EXP_ID="train_eval_score"
EXP_NAME="1_gpu_score"
mkdir -p log/${EXP_ID}_${EXP_NAME}

export CUDA_VISIBLE_DEVICES=0
python train_eval.py -id=${EXP_ID} \
                     -name=${EXP_NAME} \
                     -roberta_path="models/augment_all_context_roberta_ep3" \
                     -path_to_prepoc="data/processed_sqa" \
                     -mode=train