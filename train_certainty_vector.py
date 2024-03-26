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
import math
import wandb
import inspect
import fire
from utils import * 


datasets.disable_caching()
torch.set_grad_enabled(False)
print_sys_info()


def train_certainty_vector(model_name, 
                           train_dst_path: str, 
                           val_dst_path: str, 
                           c_metric: str, 
                           c_th: float, 
                           score_func: str, 
                           lr: float, 
                           batch_size : int, 
                           epochs: int,
                           max_train_data_size = 10000,
                           max_val_data_size = 1000,
                           label_type = 'hard'
                           ):
    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    for arg in args:
        print(f"{arg} = {values[arg]}")
    
    dst_name = val_dst_path.split("/")[-1]
    dst_name = dst_name.split("_")[1]
    if "long" in val_dst_path:
        dst_name += "_long"
        dst_type = "long"
    else:
        dst_name += "_short"
        dst_type = "short"

    print(f"dst_name:{dst_name}")
    
    # Model Config
    hooked_transformer_name = get_hooked_transformer_name(model_name)
    hf_model_path = os.path.join(os.environ["my_models_dir"], model_name)
    
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    with LoadWoInit():
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)

    model = HookedTransformer.from_pretrained_no_processing(hooked_transformer_name, dtype='bfloat16', hf_model=hf_model, tokenizer=hf_tokenizer, default_padding_side='left')

    # Data Config
    train_dst = Dataset.load_from_disk(train_dst_path)
    val_dst = Dataset.load_from_disk(val_dst_path)
    
    if max_train_data_size < len(train_dst):
        train_dst = train_dst.select(list(range(max_train_data_size)))
    if max_val_data_size < len(val_dst):
        val_dst = val_dst.select(list(range(max_val_data_size)))
    
    train_data_size = len(train_dst)
    val_data_size = len(val_dst)

    train_dst = train_dst.map(wash_answer, new_fingerprint=str(time()))
    val_dst = val_dst.map(wash_answer, new_fingerprint=str(time()))

    if c_metric == 'rougel':
        train_dst = train_dst.map(get_rougel, new_fingerprint=str(time()))
        val_dst = val_dst.map(get_rougel, new_fingerprint=str(time()))
    elif c_metric == 'sentnli':
        se_bert_name = "microsoft/deberta-large-mnli"
        se_bert_pipe = pipeline("text-classification", model=se_bert_name, device=0)
        train_dst = train_dst.map(partial(get_sentnli, se_bert_pipe=se_bert_pipe), batched=True, batch_size=2, new_fingerprint=str(time()))
        val_dst = val_dst.map(partial(get_sentnli, se_bert_pipe=se_bert_pipe), batched=True, batch_size=2, new_fingerprint=str(time()))
    elif c_metric == 'sentsim':
        st_model = SentenceTransformer('all-MiniLM-L6-v2')
        train_dst = train_dst.map(partial(get_sentsim, st_model=st_model), batched=True, batch_size=2, new_fingerprint=str(time()))
        val_dst = val_dst.map(partial(get_sentsim, st_model=st_model), batched=True, batch_size=2, new_fingerprint=str(time()))
    else:
        raise ValueError(f"metric {c_metric} not supported")
    
    keys = (['options'] if val_dst[0].get('options') else []) + ['question', 'washed_answer', 'gt', c_metric]
    for i in range(10):
        for k in keys:
            print(f"{k}:{val_dst[i][k]}")
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

    train_dst = train_dst.map(partial(get_num_tokens, tokenizer=hf_tokenizer), new_fingerprint=str(time()), batched=True, batch_size=batch_size)
    val_dst = val_dst.map(partial(get_num_tokens, tokenizer=hf_tokenizer), new_fingerprint=str(time()), batched=True, batch_size=batch_size)

    # setup optimizer
    optimizer = torch.optim.Adam(v_c.parameters(), lr=lr)

    # setup progress bar and wandb
    bar = tqdm(total=(math.ceil(len(train_dst) / batch_size) + math.ceil(len(val_dst) / batch_size)) * epochs, unit='step')
    wandb.init(project='uncertainty', name=f"{dst_name}_{c_metric}_{c_th}", tags=[score_func, model_name, label_type])
    best_auroc = 0

    def forward_func(batch) -> Float[torch.Tensor, 'b']:
        layer_batch_scores = []

        def score_hook(resid: Float[torch.Tensor, 'b p d'], hook: HookPoint):
            v_c_l = v_c[hook.name.replace(".", "#")]
            r = resid[:, -max(batch['num_answer_tokens']) - 2:, :]
            batch_all_scores = v_c_l(r)  # [b p d] -> [b p 1]
            batch_all_scores = batch_all_scores.squeeze()
            batch_scores = []
            for scores, idxs in zip(batch_all_scores, batch['answer_idxs']):
                if score_func == "sum":
                    s = scores[idxs].sum()
                elif score_func == "mean":
                    s = scores[idxs].mean()
                elif score_func == "last":
                    s = scores[idxs][-1]
                elif score_func == "max":
                    s = scores[idxs].max()
                else:
                    raise ValueError(f"score_func {score_func} not supported")
                    
                batch_scores.append(s)
            batch_scores = torch.stack(batch_scores)
            layer_batch_scores.append(batch_scores)
            return resid

        out = model.run_with_hooks(batch['washed_output'], fwd_hooks=[(lambda x: x in full_act_names, score_hook)], padding_side='left')

        batch_scores = einops.reduce(layer_batch_scores, 'l b -> b', 'mean')
        return batch_scores

    def loss_func(batch_scores, batch_labels):
        if label_type == 'hard':
            batch_labels = [1 if l > c_th else 0 for l in batch_labels]
        elif label_type == 'soft':
            pass
        else:
            raise ValueError(f"label_type {label_type} not supported")
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

            if phase == 'train':
                random.seed(42 + epoch)
                dst = train_dst.shuffle(seed=42 + epoch)
                v_c.train()
            else:
                dst = val_dst
                v_c.eval()

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
        print(f"epoch {epoch} log:{epoch_log}")
        
        if epoch_log['val_auroc'] > best_auroc:
            best_auroc = epoch_log['val_auroc']
            save_name = f"v_c_{dst_name}_{train_data_size}_{score_func}_{c_metric}_{label_type}_{model_name}_best.pth"
            torch.save(v_c.state_dict(), f"models/{save_name}")
            print(f"new best auroc:{best_auroc}")
        # plt.update(epoch_log)
        # plt.send()

    torch.set_grad_enabled(False)

    wandb.finish()

if __name__ == "__main__":
    fire.Fire(train_certainty_vector)
