#!/bin/bash

dst_names=(
    "sciq"
    "coqa"
    "triviaqa"
    "medmcqa"
    "MedQA-USMLE-4-options"
)


dst_types=(
    "long"
    "short"
)

score_funcs=(
    "mean"
    "last"
)

label_types=(
    "soft"
    # "hard"
)


log_path="/mnt/petrelfs/guoyiqiu/coding/slurm_log/%j-%x.out"
model_name="vicuna-7b-v1.1"
c_metric="include"
c_th=0.5
lr=1e-3
batch_size=16
epochs=5
max_train_data_size=2000
max_val_data_size=1000


for dst_name in "${dst_names[@]}"; do
    for dst_type in "${dst_types[@]}"; do
        train_dst_path=cached_results/"$model_name"/"$dst_type"/"$dst_name"_train
        val_dst_path=cached_results/"$model_name"/"$dst_type"/"$dst_name"_validation
        for score_func in "${score_funcs[@]}"; do
            for label_type in "${label_types[@]}"; do
                job_name=train_"$dst_name"_"$dst_type"_"$score_func"_"$label_type"
                srun --async -o $log_path -e $log_path -J $job_name -p medai --gres=gpu:1 --quotatype=spot python train_certainty_vector.py \
                    --model_name=$model_name \
                    --train_dst_path=$train_dst_path \
                    --val_dst_path=$val_dst_path \
                    --c_metric=$c_metric \
                    --c_th=$c_th \
                    --score_func=$score_func \
                    --lr=$lr \
                    --batch_size=$batch_size \
                    --epochs=$epochs \
                    --max_train_data_size=$max_train_data_size \
                    --max_val_data_size=$max_val_data_size \
                    --label_type=$label_type
                sleep 0.5
            done
        done
    done
done

sleep 2
rm batchscript*