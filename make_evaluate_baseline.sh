#!/bin/bash

log_path="/mnt/petrelfs/guoyiqiu/coding/slurm_log/%j-%x.out"
model_name="vicuna-7b-v1.1"
c_metric="all"
u_metric="pe,sar,ls,se"
custom_vc_path=""
max_val_data_size=1000

dst_names=(
    # "sciq"
    # "coqa"
    # "triviaqa"
    "medmcqa"
    "MedQA-USMLE-4-options"
)


dst_types=(
    "long"
    "short"
)

for dst_name in "${dst_names[@]}"; do
    for dst_type in "${dst_types[@]}"; do
        job_name=eval_baseline_"$dst_name"_"$dst_type"
        custom_save_path=baseline_eval_results/"$model_name"/"$dst_name"_"$dst_type"
        srun --async -o $log_path -e $log_path -J $job_name -p medai_llm --gres=gpu:1 --quotatype=spot python evaluate.py \
            --model_name $model_name \
            --dst_name $dst_name \
            --dst_type $dst_type \
            --c_metric=$c_metric \
            --u_metric=$u_metric \
            --custom_vc_path=$custom_vc_path \
            --custom_save_path=$custom_save_path \
            --max_val_data_size=$max_val_data_size \
        sleep 1
    done
done

sleep 2
rm batchscript*