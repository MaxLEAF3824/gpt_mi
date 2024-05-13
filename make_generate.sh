#!/bin/bash

conda activate mi

i=0
log_path="/mnt/petrelfs/guoyiqiu/coding/slurm_log/%j-%x.out"

model_paths=(
    "/mnt/petrelfs/guoyiqiu/coding/my_models/vicuna-7b-v1.1"
    # "/mnt/petrelfs/guoyiqiu/coding/my_models/vicuna-13b-v1.1"
    # "/mnt/petrelfs/guoyiqiu/coding/my_models/vicuna-33b-v1.3"
)

dst_names=(
    "allenai/sciq"
    # "stanfordnlp/coqa"
    # "lucadiliello/triviaqa"
    # "openlifescienceai/medmcqa"
    # "GBaker/MedQA-USMLE-4-options"
)

dst_types=(
    # "short"
    "long"
)

split_names=(
    "train"
    # "validation"
)


batch_size=16
temperature=0.5
num_beams=1

for model_path in "${model_paths[@]}"; do
    model_name=$(basename "$model_path")
    for dst_name in "${dst_names[@]}"; do
        for split_name in "${split_names[@]}"; do
            if [ $split_name = "train" ] 
            then 
                sample_num=0
                data_size=10000
            else
                sample_num=10
                data_size=1000
            fi
            for dst_type in "${dst_types[@]}"; do
                job_name=generate_"$model_name"_${dst_name//\//_}_"$split_name"_"$dst_type"
                if [ $dst_type = "long" ] 
                then 
                    max_new_tokens=256
                else
                    max_new_tokens=128
                fi
                port=$(( $RANDOM % 1000 + 29500 + i ))
                echo $port
                srun --async -o $log_path -e $log_path -J "$job_name" -p medai_llm --gres=gpu:4 --quotatype=auto accelerate launch --num_processes=4 --main_process_port $port multigpu_generate.py \
                    --model_path=$model_path \
                    --dst_name=$dst_name \
                    --max_new_tokens=$max_new_tokens \
                    --batch_size=$batch_size \
                    --data_size=$data_size \
                    --split_name=$split_name \
                    --sample_num=$sample_num \
                    --temperature=$temperature \
                    --num_beams=$num_beams \
                    --dst_type=$dst_type
                sleep 1
            done
        done
    done
done

sleep 2
rm batchscript*
