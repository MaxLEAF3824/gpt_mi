#!/bin/bash
# For Main Results and Ablation Study

log_path="/mnt/petrelfs/guoyiqiu/coding/slurm_log/%j-%x.out"
model_name="vicuna-7b-v1.1"
c_metric="all"
u_metric="ours"
max_val_data_size=1000

dst_names=(
    "sciq"
    "coqa" 
    "triviaqa"
    "medmcqa"
    "MedQA-USMLE-4-options"
)

dst_types=(
    "long"
    # "short"
)

vc_types=(
    "last_inp_soft_mlp_small"
    # "last_ans_soft_mlp_small"
    # "mean_all_soft_mlp_small"
    # "mean_ans_soft_mlp_small"
    # "attn_all_no_first_soft_mlp_small"
    # "attn_token_all_soft_mlp_small"
    "attn_token_inp_soft_mlp_small"
    # "last_ans_soft_mlp"
)

label_names=(
    "rougel"
    "sentsim"
    "include"
)

for dst_name in "${dst_names[@]}"; do
    for dst_type in "${dst_types[@]}"; do
        for label_name in "${label_names[@]}"; do
            for vc_type in "${vc_types[@]}"; do
                job_name=eval_ours_"$dst_name"_"$dst_type"_"$vc_type"
                custom_vc_path=models/"$model_name"/"$label_name"/v_c_"$dst_name"_"$dst_type"_"$vc_type".pth
                custom_save_path=ours_eval_results/"$model_name"/"$label_name"/$(basename $custom_vc_path)/"$dst_name"_"$dst_type"
                srun --async -o $log_path -e $log_path -J $job_name -p medai_llm --gres=gpu:1 --quotatype=spot python evaluate.py \
                    --model_name $model_name \
                    --dst_name $dst_name \
                    --dst_type $dst_type \
                    --c_metric=$c_metric \
                    --u_metric=$u_metric \
                    --custom_vc_path=$custom_vc_path \
                    --custom_save_path=$custom_save_path \
                    --max_val_data_size=$max_val_data_size
                sleep 1
            done
        done
    done
done

sleep 2
rm batchscript*