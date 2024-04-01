#!/bin/bash

log_path="/mnt/petrelfs/guoyiqiu/coding/slurm_log/%x-%j.out"
model_name="vicuna-7b-v1.1"

all_test_dst_path=(
    "/mnt/petrelfs/guoyiqiu/coding/trainable_uncertainty/cached_results/short/allenai_sciq_validation_1000_vicuna-7b-v1.1"
    # "/mnt/petrelfs/guoyiqiu/coding/trainable_uncertainty/cached_results/short/GBaker_MedQA-USMLE-4-options_test_1273_vicuna-7b-v1.1"
    # "/mnt/petrelfs/guoyiqiu/coding/trainable_uncertainty/cached_results/short/lucadiliello_triviaqa_validation_7785_vicuna-7b-v1.1"
    # "/mnt/petrelfs/guoyiqiu/coding/trainable_uncertainty/cached_results/short/openlifescienceai_medmcqa_validation_4183_vicuna-7b-v1.1"
    # "/mnt/petrelfs/guoyiqiu/coding/trainable_uncertainty/cached_results/short/stanfordnlp_coqa_validation_500_vicuna-7b-v1.1"
    # "/mnt/petrelfs/guoyiqiu/coding/trainable_uncertainty/cached_results/long/allenai_sciq_validation_1000_vicuna-7b-v1.1_long"
    # "/mnt/petrelfs/guoyiqiu/coding/trainable_uncertainty/cached_results/long/GBaker_MedQA-USMLE-4-options_test_1000_vicuna-7b-v1.1_long"
    # "/mnt/petrelfs/guoyiqiu/coding/trainable_uncertainty/cached_results/long/lucadiliello_triviaqa_validation_1000_vicuna-7b-v1.1_long"
    # "/mnt/petrelfs/guoyiqiu/coding/trainable_uncertainty/cached_results/long/openlifescienceai_medmcqa_validation_1000_vicuna-7b-v1.1_long"
    # "/mnt/petrelfs/guoyiqiu/coding/trainable_uncertainty/cached_results/long/stanfordnlp_coqa_validation_500_vicuna-7b-v1.1_long"
)

c_metric="rougel"
u_metric="se"
rescale=false

for test_dst_path in "${all_test_dst_path[@]}"; do
    filename=$(basename "$test_dst_path")
    job_name="eval_"$filename
    srun --async -o $log_path -e $log_path -J "$job_name" -p medai --gres=gpu:1 --quotatype=auto python evaluate.py \
        --model_name $model_name \
        --test_dst_path "$test_dst_path" \
        --c_metric=$c_metric \
        --u_metric=$u_metric \
        --rescale=$rescale
done

sleep 2
rm batchscript*