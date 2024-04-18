#!/bin/bash

log_path="/mnt/petrelfs/guoyiqiu/coding/slurm_log/%j-%x.out"
model_name="vicuna-7b-v1.1"
c_metric="all"
u_metric="ours"
max_val_data_size=1000
merge_existing_result=False
v_c_type='prob'

train_dst_names=(
    # "sciq"
    # "coqa"
    "triviaqa"
    # "medmcqa"
    # "MedQA-USMLE-4-options"
    # "all"
)

train_dst_types=(
    # "short"
    "long"
)

test_dst_names=(
    # "sciq"
    "coqa"
    # "triviaqa"
    # "medmcqa"
    # "MedQA-USMLE-4-options"
)

test_dst_types=(
    "short"
    # "long"
)

label_names=(
    # "rougel"
    # "sentsim"
    "include"
)

score_funcs=(
    "mean"
    # "last"
)

label_type="soft"

for train_dst_name in "${train_dst_names[@]}"; do
    for train_dst_type in "${train_dst_types[@]}"; do
        for test_dst_name in "${test_dst_names[@]}"; do
            for test_dst_type in "${test_dst_types[@]}"; do
                for label_name in "${label_names[@]}"; do
                    for score_func in "${score_funcs[@]}"; do
                        job_name=eval_cross_"$train_dst_name"_"$train_dst_type"_"$test_dst_name"_"$test_dst_type"_"$label_name"_"$score_func"
                        custom_vc_path=models/"$model_name"/"$label_name"/v_c_"$train_dst_name"_"$train_dst_type"_"$score_func"_"$label_type"_"$v_c_type".pth
                        custom_save_path=cross_eval_results/"$model_name"/"$label_name"/$(basename $custom_vc_path)/"$test_dst_name"_"$test_dst_type"
                        srun --async -o $log_path -e $log_path -J $job_name -p medai --gres=gpu:1 --quotatype=spot python evaluate.py \
                            --model_name $model_name \
                            --dst_name $test_dst_name \
                            --dst_type $test_dst_type \
                            --c_metric=$c_metric \
                            --u_metric=$u_metric \
                            --max_val_data_size=$max_val_data_size \
                            --custom_vc_path=$custom_vc_path \
                            --custom_save_path=$custom_save_path \
                            --merge_existing_result=$merge_existing_result \
                            --v_c_type=$v_c_type
                        sleep 1
                    done
                done
            done
        done
    done
done

sleep 2
rm batchscript*