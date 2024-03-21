import math
import os

import fire

os.environ['HF_DATASETS_OFFLINE'] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import transformer_lens
import datasets
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
import math
import os

os.environ['HF_DATASETS_OFFLINE'] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import transformer_lens
import datasets
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix
import einops
from fancy_einsum import einsum
from tqdm.auto import tqdm
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, Features, Array2D, Array3D
from typing import List, Tuple, Union
import os
import random
import numpy as np
from rouge import Rouge
from time import time
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from copy import deepcopy
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util
from transformers import pipeline
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
import math
import wandb


datasets.disable_caching()
torch.set_grad_enabled(False)


def print_sys_info():
    import psutil
    import socket
    import gpustat
    memory = psutil.virtual_memory()
    print("剩余内存: {} G".format(memory.available / 1024 / 1024 // 1024))
    host_name = socket.gethostname()
    print(f"当前主机名是:{host_name}")
    gpustat.print_gpustat()

print_sys_info()

class LoadWoInit:
        """Context manager that disable parameter initialization."""

        def __init__(self):
            self.constant_ = torch.nn.init.constant_
            self.zeros_ = torch.nn.init.zeros_
            self.ones_ = torch.nn.init.ones_
            self.uniform_ = torch.nn.init.uniform_
            self.normal_ = torch.nn.init.normal_
            self.kaiming_uniform_ = torch.nn.init.kaiming_uniform_
            self.kaiming_normal_ = torch.nn.init.kaiming_normal_

        def __enter__(self, *args, **kwargs):
            torch.nn.init.constant_ = lambda *args, **kwargs: None
            torch.nn.init.zeros_ = lambda *args, **kwargs: None
            torch.nn.init.ones_ = lambda *args, **kwargs: None
            torch.nn.init.uniform_ = lambda *args, **kwargs: None
            torch.nn.init.normal_ = lambda *args, **kwargs: None
            torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
            torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None

        def __exit__(self, *args, **kwargs):
            torch.nn.init.constant_ = self.constant_
            torch.nn.init.zeros_ = self.zeros_
            torch.nn.init.ones_ = self.ones_
            torch.nn.init.uniform_ = self.uniform_
            torch.nn.init.normal_ = self.normal_
            torch.nn.init.kaiming_uniform_ = self.kaiming_uniform_
            torch.nn.init.kaiming_normal_ = self.kaiming_normal_

def train_certainty_vector(model_name, train_dst_path: str, val_dst_path: str, c_metric: str, c_th: float, lr=1e-3, batch_size=32, epochs=5):
    print(f"train_dst_path:{train_dst_path}")
    print(f"val_dst_path:{val_dst_path}")
    dst_name = val_dst_path.split("/")[-1]
    print(f"dst_name:{dst_name}")
    # Model Config
    # model_name = "vicuna-7b-v1.1"
    hooked_transformer_name = "llama-7b-hf"
    
    hf_model_path = os.path.join(os.environ["my_models_dir"], model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    with LoadWoInit():
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)

    model = HookedTransformer.from_pretrained_no_processing(hooked_transformer_name, dtype='bfloat16', hf_model=hf_model, tokenizer=hf_tokenizer, default_padding_side='left')

    def wash(text):
        for sp_tok in model.tokenizer.special_tokens_map.values():
            text = text.replace(sp_tok, "")
        first_string_before_question = text
        spliters = ['question:', 'context:']
        for spliter in spliters:
            if spliter in text.lower():
                first_string_before_question = text.lower().split(spliter)[0]
                break
        text = text[:len(first_string_before_question)]
        text = text.strip()
        return text

    def wash_answer(example):
        example['washed_answer'] = wash(example['answer'])
        example['washed_output'] = example['input'] + example['washed_answer']
        if example.get("sampled_answer"):
            example['washed_sampled_answer'] = [wash(ans) for ans in example['sampled_answer']]
            example['washed_sampled_output'] = [example['input'] + ans for ans in example['washed_sampled_answer']]
        return example

    def get_rougel(example):
        rouge = Rouge()
        hyp = example['washed_answer'].lower()
        if hyp == "" or hyp == '.' or hyp == '...':
            hyp = "-"
        ref = example['gt'].lower()
        scores = rouge.get_scores(hyp, ref)
        example["rougel"] = scores[0]['rouge-l']['f']
        return example

    train_dst = Dataset.load_from_disk(train_dst_path)
    val_dst = Dataset.load_from_disk(val_dst_path).select(range(1000))

    train_dst = train_dst.map(wash_answer, new_fingerprint=str(time()))
    val_dst = val_dst.map(wash_answer, new_fingerprint=str(time()))

    train_dst = train_dst.map(get_rougel, new_fingerprint=str(time()))
    val_dst = val_dst.map(get_rougel, new_fingerprint=str(time()))

    keys = (['options'] if val_dst[0].get('options') else []) + ['question', 'answer', 'washed_answer', 'washed_sampled_answer', 'gt']

    for i in range(10):
        for k in keys:
            print(f"{k}:{val_dst[i * 10][k]}")
        print()

    torch.set_grad_enabled(True)
    model.requires_grad_(False)

    layers = list(range(0, model.cfg.n_layers))
    act_name = 'resid_post'
    full_act_names = [utils.get_act_name(act_name, l) for l in sorted(layers)]

    # module config
    v_c = nn.ModuleDict({
        act_name.replace(".", "#"): nn.Sequential(
            nn.Linear(model.cfg.d_model, model.cfg.d_model),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(model.cfg.d_model, 1),
            nn.Sigmoid()
        )
        for act_name in full_act_names
    })
    v_c.to(model.cfg.dtype).to(model.cfg.device)


    # preprocess data
    def preprocess(examples):
        batch_num_input_tokens = list(map(len, model.tokenizer(examples['input'])['input_ids']))
        batch_num_output_tokens = list(map(len, model.tokenizer(examples['washed_output'])['input_ids']))
        batch_num_answer_tokens = [num_output_tokens - num_input_tokens for num_input_tokens, num_output_tokens in zip(batch_num_input_tokens, batch_num_output_tokens)]
        batch_answer_idxs = [list(range(-num_answer_tokens - 1, 0)) for num_answer_tokens in batch_num_answer_tokens]
        examples['num_input_tokens'] = batch_num_input_tokens
        examples['num_output_tokens'] = batch_num_output_tokens
        examples['num_answer_tokens'] = batch_num_answer_tokens
        examples['answer_idxs'] = batch_answer_idxs
        return examples

    train_dst = train_dst.map(preprocess, new_fingerprint=str(time()), batched=True, batch_size=batch_size)
    val_dst = val_dst.map(preprocess, new_fingerprint=str(time()), batched=True, batch_size=batch_size)

    # setup optimizer
    optimizer = torch.optim.Adam(v_c.parameters(), lr=lr)

    # setup progress bar and plot
    bar = tqdm(total=(math.ceil(len(train_dst) / batch_size) + math.ceil(len(val_dst) / batch_size)) * epochs, unit='step')
    wandb.init(project='uncertainty')
    # plt = PlotLosses(groups={'auroc': ['train_auroc', 'val_auroc'], 'loss': ['train_loss', 'val_loss']}, outputs=[MatplotlibPlot()])
    best_auroc = 0

    def forward_func(batch) -> Float[torch.Tensor, 'b']:
        layer_batch_scores = []

        def score_hook(resid: Float[torch.Tensor, 'b p d'], hook: HookPoint):
            v_c_l = v_c[hook.name.replace(".", "#")]
            r = resid[:, -max(batch['num_answer_tokens']) - 2:, :]
            batch_all_scores = v_c_l(r)  # [b p d] -> [b p 1]
            batch_scores = torch.stack([scores[idxs].sum() for scores, idxs in zip(batch_all_scores, batch['answer_idxs'])])
            # r = resid[:, [-1], :]  # [b 1 d]
            # batch_all_scores = v_c_l(r) # [b p d] -> [b p 1]
            # batch_scores = batch_all_scores[:, -1, 0]
            layer_batch_scores.append(batch_scores)
            return resid

        out = model.run_with_hooks(batch['washed_output'], fwd_hooks=[(lambda x: x in full_act_names, score_hook)], padding_side='left')

        batch_scores = einops.reduce(layer_batch_scores, 'l b -> b', 'mean')
        return batch_scores

    def loss_func(batch_scores, batch_labels):
        batch_labels = [1 if l > c_th else 0 for l in batch_labels]
        batch_labels = torch.tensor(batch_labels, dtype=batch_scores.dtype).to(batch_scores.device)
        return F.mse_loss(batch_scores, batch_labels)

    def eval_func(scores, labels):
        discrete_labels = [1 if l > c_th else 0 for l in labels]
        return roc_auc_score(discrete_labels, scores)

    for epoch in range(epochs):
        epoch_log = {}
        for phase in ['train', 'val']:
            epoch_loss = []
            epoch_scores = []

            v_c.train() if phase == 'train' else v_c.eval()

            if phase == 'train':
                random.seed(42 + epoch)
                dst = train_dst.shuffle(seed=42 + epoch)
            else:
                dst = val_dst

            for i in range(0, len(dst), batch_size):
                batch = dst[i:i + batch_size]

                if phase == 'train':
                    batch_scores = forward_func(batch)
                else:
                    with torch.no_grad():
                        batch_scores = forward_func(batch)

                loss = loss_func(batch_scores, batch[c_metric])

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss.append(loss.item())
                epoch_scores.extend(batch_scores.tolist())

                bar.update(1)
                bar.set_description(f'{phase} loss:{loss.item():.4f}', refresh=True)

            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            epoch_auroc = eval_func(epoch_scores, dst[c_metric])
            epoch_log.update({f'{phase}_loss': epoch_loss, f'{phase}_auroc': epoch_auroc})

        wandb.log(epoch_log)
        print(epoch_log)
        
        if epoch_log['val_auroc'] > best_auroc:
            best_auroc = epoch_log['val_auroc']
            torch.save(v_c.state_dict(), f"v_c_{dst_name}_{c_metric}_{model_name}_best.pth")
        # plt.update(epoch_log)
        # plt.send()

    torch.set_grad_enabled(False)

    wandb.finish()

if __name__ == "__main__":
    fire.Fire(train_certainty_vector)
