from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch
import time
import json
from tqdm import tqdm
import datasets
import random
import os
import fire
import inspect
from utils import LoadWoInit, print_sys_info
from functools import partial

os.environ['HF_DATASETS_OFFLINE'] = "1"
accelerator = Accelerator()
print_sys_info()


def main(
        model_path,
        dst_name,
        split_name,
        data_size,
        sample_num,
        temperature,
        num_beams,
        add_vicuna_prompt,
        max_new_tokens,
        batch_size
):
    model_name = model_path.split('/')[-1]

    if dst_name == 'GBaker/MedQA-USMLE-4-options':
        dst = datasets.load_dataset("/mnt/petrelfs/guoyiqiu/coding/huggingface/datasets/GBaker___med_qa-usmle-4-options/default/0.0.0/0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18")
    else:
        dst = datasets.load_dataset(dst_name)

    def preprocess_sciq(example, idx):
        dst_template = "Question:{q} Options:{o} Answer:"
        options = [example["distractor1"], example["distractor2"], example["distractor3"], example["correct_answer"]]
        random.seed(42 + idx)
        random.shuffle(options)
        input = dst_template.format(q=example["question"], o=", ".join(options))
        example['input'] = input
        example['dst_template'] = dst_template
        example['options'] = options
        example['gt'] = example["correct_answer"]
        return example

    def preprocess_coqa(example, idx):
        max_num_shots = 1
        qa_template = "Question:{q} Answer:{a}"
        dst_template = "Context:{c} {qa} Question:{q} Answer:"
        example['dst_template'] = dst_template
        questions = example['questions'][:max_num_shots]
        answers = example['answers']['input_text'][:max_num_shots]
        qa = " ".join([qa_template.format(q=q, a=a) for q, a in zip(questions[:-1], answers[:-1])])
        question = questions[-1]
        example['question'] = question
        gt = answers[-1]
        input = dst_template.format(c=example['story'], qa=qa, q=question)
        example['input'] = input
        example['gt'] = gt
        return example

    def preprocess_triviaqa(example, idx, dst):
        num_shots = 10
        dst_template = "Question:{q} Answer:{a}"
        example['dst_template'] = dst_template
        shots = dst.select(range(num_shots))
        shots = list(map(lambda x: dst_template.format(q=x['question'], a=', '.join(x['answers'])), shots))
        shots = " ".join(shots)
        cur_shot = dst_template.format(q=example['question'], a='')
        input = f"{shots} {cur_shot}"
        example['input'] = input
        example['gt'] = ', '.join(example['answers'])
        return example

    def preprocess_medmcqa(example, idx):
        dst_template = "Question:{q} Options:{o} Answer:"
        options = [example["opa"], example["opb"], example["opc"], example["opd"]]
        input = dst_template.format(q=example["question"], o=", ".join(options))
        example['input'] = input
        example['dst_template'] = dst_template
        example['options'] = options
        example['gt'] = options[example['cop']]
        return example

    def preprocess_medqa(example, idx):
        dst_template = "Question:{q} Options:{o} Answer:"
        options = list(example["options"].values())
        random.seed(42 + idx)
        random.shuffle(options)
        input = dst_template.format(q=example["question"], o=", ".join(options))
        example['input'] = input
        example['dst_template'] = dst_template
        example['options'] = options
        example['gt'] = example["answer"]
        del example['answer']
        return example

    def add_vicuna_prompt_to_input(example):
        vicuna_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER:{} ASSISTANT:"
        example['input'] = vicuna_prompt.format(example['input'])
        return example

    preprocess_triviaqa = partial(preprocess_triviaqa, dst=dst['train'])

    preprocess_map = {
        "allenai/sciq": preprocess_sciq,
        "stanfordnlp/coqa": preprocess_coqa,
        "lucadiliello/triviaqa": preprocess_triviaqa,
        "openlifescienceai/medmcqa": preprocess_medmcqa,
        "GBaker/MedQA-USMLE-4-options": preprocess_medqa
    }

    if split_name not in ['train', 'validation']:
        raise ValueError("split_name must be one of ['train','validation','test']")
    if split_name == 'validation':
        if dst.get('validation'):
            dst = dst['validation']
        elif dst.get('test'):
            dst = dst['test']
        else:
            raise ValueError(f"dst {dst_name} not have a validation of test split")
    else:
        dst = dst[split_name]
    if data_size is not None and data_size <= len(dst):
        dst = dst.select(range(data_size))
    else:
        data_size = len(dst)
    preprocess_func = preprocess_map[dst_name]
    dst = dst.map(preprocess_func, with_indices=True, new_fingerprint=str(time.time()))
    if add_vicuna_prompt:
        dst = dst.map(add_vicuna_prompt_to_input, new_fingerprint=str(time.time()))
    prompts_all = dst['input']

    # load a base model and tokenizer
    with LoadWoInit():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": accelerator.process_index},
            torch_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # batch, left pad (for inference), and tokenize
    def prepare_prompts(prompts, tokenizer, batch_size=16):
        batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
        batches_tok = []
        tokenizer.padding_side = "left"
        for prompt_batch in batches:
            batches_tok.append(
                tokenizer(
                    prompt_batch,
                    return_tensors="pt",
                    padding=True,
                ).to("cuda")
            )
        tokenizer.padding_side = "right"
        return batches_tok

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start = time.time()
    if accelerator.is_main_process:
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        for arg in args:
            print(f"{arg} = {values[arg]}")
        print(f"Start generating {len(prompts_all)} prompts on {accelerator.num_processes} GPUs")

    # divide the prompt list onto the available GPUs
    with accelerator.split_between_processes(prompts_all) as prompts:
        results = dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches = prepare_prompts(prompts, tokenizer, batch_size=batch_size)

        for prompts_tokenized in tqdm(prompt_batches):
            outputs_tokenized = model.generate(**prompts_tokenized, max_new_tokens=max_new_tokens, do_sample=False)
            # remove prompt from gen. tokens
            outputs_tokenized = [tok_out[len(tok_in):]
                                 for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)]
            # count and decode gen. tokens
            num_tokens = sum([len(t) for t in outputs_tokenized])
            outputs = tokenizer.batch_decode(outputs_tokenized)

            # store in results{} to be gathered by accelerate
            results["outputs"].extend(outputs)
            results["num_tokens"] += num_tokens

        if accelerator.is_main_process and sample_num > 0:
            print("Start generating Sample outputs...")
        for sample_idx in tqdm(range(sample_num)):
            results[f"sampled_outputs_{sample_idx}"] = []
            for prompts_tokenized in prompt_batches:
                outputs_tokenized = model.generate(**prompts_tokenized, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, num_beams=num_beams)
                # remove prompt from gen. tokens
                outputs_tokenized = [tok_out[len(tok_in):]
                                     for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)]
                # count and decode gen. tokens
                num_tokens = sum([len(t) for t in outputs_tokenized])
                outputs = tokenizer.batch_decode(outputs_tokenized)

                # store in results{} to be gathered by accelerate
                results[f"sampled_outputs_{sample_idx}"].extend(outputs)
                results["num_tokens"] += num_tokens

        results = [results]  # transform to list, otherwise gather_object() will not collect correctly

    # collect results from all the GPUs
    results_gathered = gather_object(results)

    if accelerator.is_main_process:
        timediff = time.time() - start
        num_tokens = sum([r["num_tokens"] for r in results_gathered])
        print(f"tokens/sec: {num_tokens // timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")

        all_answers = []
        for result_i in results_gathered:
            all_answers.extend(result_i['outputs'])

        all_outputs = [inp + ans for inp, ans in zip(dst['input'], all_answers)]
        dst = dst.add_column("answer", all_answers)
        dst = dst.add_column("output", all_outputs)

        if sample_num > 0:
            all_sampled_answers_group = [[] for i in range(sample_num)]
            for result_i in results_gathered:
                for sample_idx in range(sample_num):
                    all_sampled_answers_group[sample_idx].extend(result_i[f"sampled_outputs_{sample_idx}"])

            all_sampled_answers = [[g[i] for g in all_sampled_answers_group] for i in range(len(all_answers))]
            all_sampled_outputs = [[dst['input'][i] + g[i] for g in all_sampled_answers_group] for i in range(len(all_answers))]
            dst = dst.add_column("sampled_answer", all_sampled_answers)
            dst = dst.add_column("sampled_output", all_sampled_outputs)

        save_name = f"{dst_name.replace('/', '_')}_{split_name}_{data_size}"
        save_path = f"cached_results/{model_name}/{'long/' if add_vicuna_prompt else ''}{save_name}"
        dst.save_to_disk(save_path)
        print(f"Result Saved to {save_path}")


if __name__ == "__main__":
    fire.Fire(main)
