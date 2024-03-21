#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --partition=medai
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=spot
#SBATCH --output=slurm_log/%x-%j.out
#SBATCH --error=slurm_log/%x-%j.out


# dst_name="allenai/sciq"
# dst_name="stanfordnlp/coqa"
# dst_name="lucadiliello/triviaqa"
# dst_name="openlifescienceai/medmcqa"
# dst_name="GBaker/MedQA-USMLE-4-options"

# accelerate launch --num_processes=4 --main_process_port 29567 multigpu_generate.py \
#     --model_path="/mnt/petrelfs/guoyiqiu/coding/my_models/vicuna-7b-v1.1" \
#     --dst_name="allenai/sciq" \
#     --max_new_tokens=256 \
#     --batch_size=16 \
#     --data_size=1000 \
#     --split_name='validation' \
#     --sample_num=10 \
#     --temperature=0.5 \
#     --num_beams=1 \
#     --add_vicuna_prompt=True

# accelerate launch --num_processes=4 --main_process_port 29567 multigpu_generate.py \
#     --model_path="/mnt/petrelfs/guoyiqiu/coding/my_models/vicuna-7b-v1.1" \
#     --dst_name="stanfordnlp/coqa" \
#     --max_new_tokens=256 \
#     --batch_size=16 \
#     --data_size=10000 \
#     --split_name='train' \
#     --sample_num=10 \
#     --temperature=0.5 \
#     --num_beams=1 \
#     --add_vicuna_prompt=True

# accelerate launch --num_processes=4 --main_process_port 29567 multigpu_generate.py \
#     --model_path="/mnt/petrelfs/guoyiqiu/coding/my_models/vicuna-7b-v1.1" \
#     --dst_name="stanfordnlp/coqa" \
#     --max_new_tokens=256 \
#     --batch_size=16 \
#     --data_size=1000 \
#     --split_name='validation' \
#     --sample_num=10 \
#     --temperature=0.5 \
#     --num_beams=1 \
#     --add_vicuna_prompt=True

# accelerate launch --num_processes=4 --main_process_port 29567 multigpu_generate.py \
#     --model_path="/mnt/petrelfs/guoyiqiu/coding/my_models/vicuna-7b-v1.1" \
#     --dst_name="lucadiliello/triviaqa" \
#     --max_new_tokens=256 \
#     --batch_size=16 \
#     --data_size=10000 \
#     --split_name='train' \
#     --sample_num=10 \
#     --temperature=0.5 \
#     --num_beams=1 \
#     --add_vicuna_prompt=True

# accelerate launch --num_processes=4 --main_process_port 29567 multigpu_generate.py \
#     --model_path="/mnt/petrelfs/guoyiqiu/coding/my_models/vicuna-7b-v1.1" \
#     --dst_name="lucadiliello/triviaqa" \
#     --max_new_tokens=256 \
#     --batch_size=16 \
#     --data_size=1000 \
#     --split_name='validation' \
#     --sample_num=10 \
#     --temperature=0.5 \
#     --num_beams=1 \
#     --add_vicuna_prompt=True

# accelerate launch --num_processes=4 --main_process_port 29567 multigpu_generate.py \
#     --model_path="/mnt/petrelfs/guoyiqiu/coding/my_models/vicuna-7b-v1.1" \
#     --dst_name="openlifescienceai/medmcqa" \
#     --max_new_tokens=256 \
#     --batch_size=16 \
#     --data_size=10000 \
#     --split_name='train' \
#     --sample_num=10 \
#     --temperature=0.5 \
#     --num_beams=1 \
#     --add_vicuna_prompt=True

# accelerate launch --num_processes=4 --main_process_port 29567 multigpu_generate.py \
#     --model_path="/mnt/petrelfs/guoyiqiu/coding/my_models/vicuna-7b-v1.1" \
#     --dst_name="openlifescienceai/medmcqa" \
#     --max_new_tokens=256 \
#     --batch_size=16 \
#     --data_size=1000 \
#     --split_name='validation' \
#     --sample_num=10 \
#     --temperature=0.5 \
#     --num_beams=1 \
#     --add_vicuna_prompt=True

# accelerate launch --num_processes=4 --main_process_port 29567 multigpu_generate.py \
#     --model_path="/mnt/petrelfs/guoyiqiu/coding/my_models/vicuna-7b-v1.1" \
#     --dst_name="GBaker/MedQA-USMLE-4-options" \
#     --max_new_tokens=256 \
#     --batch_size=16 \
#     --data_size=10000 \
#     --split_name='train' \
#     --sample_num=10 \
#     --temperature=0.5 \
#     --num_beams=1 \
#     --add_vicuna_prompt=True
#
# accelerate launch --num_processes=4 --main_process_port 29567 multigpu_generate.py \
#     --model_path="/mnt/petrelfs/guoyiqiu/coding/my_models/vicuna-7b-v1.1" \
#     --dst_name="GBaker/MedQA-USMLE-4-options" \
#     --max_new_tokens=256 \
#     --batch_size=16 \
#     --data_size=1000 \
#     --split_name='test' \
#     --sample_num=10 \
#     --temperature=0.5 \
#     --num_beams=1 \
#     --add_vicuna_prompt=True

accelerate launch --num_processes=4 --main_process_port 29567 multigpu_generate.py \
    --model_path="/mnt/petrelfs/guoyiqiu/coding/my_models/vicuna-7b-v1.1" \
    --dst_name="lucadiliello/triviaqa" \
    --max_new_tokens=128 \
    --batch_size=16 \
    --data_size=10000 \
    --split_name='train' \
    --sample_num=10 \
    --temperature=0.5 \
    --num_beams=1 \
    --add_vicuna_prompt=False
