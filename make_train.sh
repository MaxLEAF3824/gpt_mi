#!/bin/bash
#SBATCH --job-name=train_v_c
#SBATCH --partition=medai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=auto
#SBATCH --output=slurm_log/%x-%j.out
#SBATCH --error=slurm_log/%x-%j.out

python train_certainty_vector.py \
    --model_name="vicuna-7b-v1.1" \
    --train_dst_path="/mnt/petrelfs/guoyiqiu/coding/gpt_mi/cached_results/lucadiliello_triviaqa_train_10000_vicuna-7b-v1.1" \
    --val_dst_path="/mnt/petrelfs/guoyiqiu/coding/gpt_mi/cached_results/lucadiliello_triviaqa_validation_7785_vicuna-7b-v1.1" \
    --c_metric="rougel" \
    --c_th=0.5 \
    --lr=1e-3 \
    --batch_size=32 \
    --epochs=5

python train_certainty_vector.py \
    --model_name="vicuna-7b-v1.1" \
    --train_dst_path="/mnt/petrelfs/guoyiqiu/coding/gpt_mi/cached_results/allenai_sciq_train_2000_vicuna-7b-v1.1" \
    --val_dst_path="/mnt/petrelfs/guoyiqiu/coding/gpt_mi/cached_results/allenai_sciq_validation_1000_vicuna-7b-v1.1" \
    --c_metric="rougel" \
    --c_th=0.5 \
    --lr=1e-3 \
    --batch_size=32 \
    --epochs=5

python train_certainty_vector.py \
    --model_name="vicuna-7b-v1.1" \
    --train_dst_path="/mnt/petrelfs/guoyiqiu/coding/gpt_mi/cached_results/allenai_sciq_train_11679_vicuna-7b-v1.1_long" \
    --val_dst_path="/mnt/petrelfs/guoyiqiu/coding/gpt_mi/cached_results/allenai_sciq_validation_1000_vicuna-7b-v1.1_long" \
    --c_metric="rougel" \
    --c_th=0.5 \
    --lr=1e-3 \
    --batch_size=32 \
    --epochs=5