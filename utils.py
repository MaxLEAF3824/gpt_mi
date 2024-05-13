from sentence_transformers.util import cos_sim
import fire
from transformers import pipeline
from sentence_transformers import util as st_util
from sentence_transformers import SentenceTransformer
import re
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from time import time
from rouge import Rouge
import numpy as np
import random
from typing import List, Tuple, Union
from datasets import load_dataset, Dataset, Features, Array2D, Array3D
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from functools import partial
from jaxtyping import Float
import torch.nn.functional as F
import torch.nn as nn
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly
from tqdm.auto import tqdm
from fancy_einsum import einsum
import einops
from transformer_lens import HookedTransformer, FactoredMatrix
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
import transformer_lens.utils as utils
import pandas as pd
import datasets
import transformer_lens
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Union

os.environ['HF_DATASETS_OFFLINE'] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Timer:
    def __enter__(self):
        self.ts = time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.te = time()
        self.t = self.te - self.ts


# Fast Load Model Context Manager
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


def print_sys_info():
    import psutil
    import socket
    import gpustat
    memory = psutil.virtual_memory()
    print("剩余内存: {} G".format(memory.available / 1024 / 1024 // 1024))
    host_name = socket.gethostname()
    print(f"当前主机名是:{host_name}")
    gpustat.print_gpustat()


def get_hooked_transformer_name(model_name):
    if "7b" in model_name:
        hooked_transformer_name = "llama-7b-hf"
    elif "13b" in model_name:
        hooked_transformer_name = "llama-13b-hf"
    elif "30b" in model_name or "33b" in model_name:
        hooked_transformer_name = "llama-30b-hf"
    else:
        hooked_transformer_name = "llama-7b-hf"
    return hooked_transformer_name


# Preprocess Function
def find_sequence_positions(list1, list2):
    sequence_length = len(list1)
    for i in range(len(list2) - sequence_length + 1):
        if list2[i:i + sequence_length] == list1:
            return i


def wash(answer_ids, tokenizer):
    # ['.', '\n', 'Question:', 'Context:', 'Options:'] Miserable hard-coded due to the fking bug of llama tokenizer
    custom_sp_ids = [[869], [29889], [13], [16492, 29901], [2677, 29901], [15228, 29901], [894, 29901], [25186, 29901]]
    new_answer_ids = deepcopy(answer_ids)
    special_ids = [[i] for i in tokenizer.all_special_ids]
    for sp_id in special_ids + custom_sp_ids:
        if len(sp_id) == 1:
            if sp_id[0] in new_answer_ids:
                new_answer_ids = new_answer_ids[:new_answer_ids.index(sp_id[0])]
        else:
            idx = find_sequence_positions(sp_id, new_answer_ids)
            new_answer_ids = new_answer_ids[:idx]
    return new_answer_ids


def wash_answer(example, tokenizer):
    example['washed_answer_ids'] = wash(example['answer_ids'], tokenizer)
    example['washed_answer'] = tokenizer.decode(example['washed_answer_ids'], skip_special_tokens=True)
    if example.get("sampled_answer"):
        example['washed_sampled_answer_ids'] = [wash(ans, tokenizer) for ans in example['sampled_answer_ids']]
        example['washed_sampled_answer'] = [tokenizer.decode(ans, skip_special_tokens=True) for ans in example['washed_sampled_answer_ids']]
    return example


def get_rougel(example):
    rouge = Rouge()
    hyp = example['washed_answer'].lower()
    if hyp.strip() == "" or hyp == '.' or hyp == '...':
        hyp = "#"
    ref = example['gt'].lower()
    scores = rouge.get_scores(hyp, ref)
    example["rougel"] = scores[0]['rouge-l']['f']
    return example


def get_sentnli(examples, nli_pipe):
    bsz = len(examples['input'])
    batch_nli_input = []
    batch_sentnli = []
    for i in range(bsz):
        example = {k: examples[k][i] for k in examples.keys()}
        nli_tmp = "[CLS] {s1} [SEP] {s2} [CLS]"
        qa_tmp = "Question:{q} Answer:{a}"
        s1 = qa_tmp.format(q=example['question'], a=example['gt'])
        s2 = qa_tmp.format(q=example['question'], a=example['washed_answer'])
        batch_nli_input.extend([nli_tmp.format(s1=s1, s2=s2), nli_tmp.format(s1=s2, s2=s1)])
    res = nli_pipe(batch_nli_input)

    for i in range(0, bsz * 2, 2):
        score = 0.
        if res[i]['label'] == 'ENTAILMENT':
            score += 0.5
        if res[i + 1]['label'] == 'ENTAILMENT':
            score += 0.5
        batch_sentnli.append(score)
    examples['sentnli'] = batch_sentnli
    return examples


def get_sentsim(examples, st_model):
    sentences1 = examples['washed_answer']
    sentences2 = examples['gt']
    embeddings1 = st_model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = st_model.encode(sentences2, convert_to_tensor=True)
    sim = torch.diag(cos_sim(embeddings1, embeddings2))
    # sim = (sim + 1) / 2
    batch_sentsim = sim.tolist()
    examples['sentsim'] = batch_sentsim
    return examples


def get_include(example):
    wrong_answers = []
    if example.get('options', []):
        wrong_answers = [o for o in example['options'] if o != example['gt']]
    include = 0
    if example['gt'].lower() in example['washed_answer'].lower():
        include = 1
        for wa in wrong_answers:
            if wa.lower() in example['washed_answer'].lower():
                include = 0
                break
    example['include'] = include
    return example


def get_num_tokens(examples):
    batch_num_input_tokens = list(map(len, examples['input_ids']))
    batch_num_answer_tokens = list(map(len, examples['washed_answer_ids']))
    batch_num_output_tokens = [len(i + a) for i, a in zip(examples['input_ids'], examples['washed_answer_ids'])]
    batch_answer_idxs = [list(range(-num_answer_tokens - 1, -1)) for num_answer_tokens in batch_num_answer_tokens]
    examples['num_input_tokens'] = batch_num_input_tokens
    examples['num_output_tokens'] = batch_num_output_tokens
    examples['num_answer_tokens'] = batch_num_answer_tokens
    examples['answer_idxs'] = batch_answer_idxs
    return examples


def _get_answer_prob(input_ids, washed_answer_ids, prob):
    if len(washed_answer_ids) == 0:
        return []
    prob = prob[len(input_ids) - 1:len(input_ids) + len(washed_answer_ids) - 1]
    answer_prob = prob[range(len(washed_answer_ids)), washed_answer_ids]
    return answer_prob.tolist()


def _get_batch_padded_output_ids(batch_input_ids, batch_washed_answer_ids, pad_token_id, padding_side):
    batch_output_ids = [torch.tensor(inp + ans, dtype=torch.long) for inp, ans in zip(batch_input_ids, batch_washed_answer_ids)]
    mask = [torch.ones_like(o) for o in batch_output_ids]
    max_len = max(map(len, batch_output_ids))
    if padding_side == "right":
        padded_output_ids = torch.cat([F.pad(output_ids, (0, max_len - len(output_ids)), value=pad_token_id).unsqueeze(0) for output_ids in batch_output_ids], dim=0)
        mask = torch.cat([F.pad(m, (0, max_len - len(m)), value=0).unsqueeze(0) for m in mask], dim=0)
    elif padding_side == "left":
        padded_output_ids = torch.cat([F.pad(output_ids, (max_len - len(output_ids), 0), value=pad_token_id).unsqueeze(0) for output_ids in batch_output_ids], dim=0)
        mask = torch.cat([F.pad(m, (max_len - len(m), 0), value=0).unsqueeze(0) for m in mask], dim=0)
    else:
        raise ValueError(f"padding_side {padding_side} not supported")

    return padded_output_ids, mask


def get_answer_prob(examples, model):
    bsz = len(examples['input'])
    batch_answer_prob = []
    padded_output_ids, mask = _get_batch_padded_output_ids(examples['input_ids'], examples['washed_answer_ids'], model.tokenizer.pad_token_id, 'right')
    with Timer() as timer:
        batch_prob = F.softmax(model(padded_output_ids, prepend_bos=False, padding_side='right'), dim=-1)  # prob: (bsz pos vocab)
    for i in range(bsz):
        batch_answer_prob.append(_get_answer_prob(examples['input_ids'][i], examples['washed_answer_ids'][i], batch_prob[i]))

    examples['answer_prob'] = batch_answer_prob
    examples['time_fwd'] = [timer.t / bsz for i in range(bsz)]
    return examples


def get_sampled_answer_prob(example, model):
    batch_answer_prob = []
    washed_sampled_answer_ids = [tuple(i) for i in example['washed_sampled_answer_ids']]
    uni_ids = list(set(washed_sampled_answer_ids))
    padded_output_ids, mask = _get_batch_padded_output_ids([example['input_ids']] * len(uni_ids), list(map(list, uni_ids)), model.tokenizer.pad_token_id, 'right')
    batch_prob = F.softmax(model(padded_output_ids, prepend_bos=False, padding_side='right'), dim=-1)  # logits: (bsz pos vocab)

    for i in range(len(washed_sampled_answer_ids)):
        prob = batch_prob[uni_ids.index(washed_sampled_answer_ids[i])]
        batch_answer_prob.append(_get_answer_prob(example['input_ids'], example['washed_sampled_answer_ids'][i], prob))

    example['sampled_answer_prob'] = batch_answer_prob
    return example


# Uncertainty Estimation Baselines
def get_uncertainty_score_token_pe_all(examples, model):
    with Timer() as timer:
        bsz = len(examples['input'])
        examples['u_score_pe_all'] = []
        examples['u_score_pe'] = []
        examples['u_score_ln_pe'] = []
        for i in range(bsz):
            if not examples['answer_prob'][i]:
                examples['u_score_pe_all'].append([])
                examples['u_score_pe'].append(0)
                examples['u_score_ln_pe'].append(0)
                continue
            neglogp = -torch.log(torch.tensor(examples['answer_prob'][i], dtype=torch.float))
            examples['u_score_pe'].append(neglogp.sum().item())
            examples['u_score_pe_all'].append(neglogp.tolist())
            examples['u_score_ln_pe'].append(neglogp.mean().item())
    examples['time_pe'] = [timer.t / bsz for i in range(bsz)]
    return examples


def get_uncertainty_score_ls(example):
    with Timer() as timer:
        # Sample Answers
        sampled_answers = example['washed_sampled_answer']
        rouge = Rouge()
        hyps = []
        refs = []
        for i in range(len(sampled_answers)):
            for j in range(i + 1, len(sampled_answers)):
                hyp = sampled_answers[i]
                ref = sampled_answers[j]
                if hyp == "" or hyp == '.':
                    hyp = "-"
                if ref == "" or ref == '.':
                    ref = "-"
                hyps.append(hyp)
                refs.append(ref)
        scores = rouge.get_scores(hyps, refs, avg=True)
        example['u_score_ls'] = 1 - scores['rouge-l']['f']
    example['time_ls'] = timer.t
    return example


def get_uncertainty_score_se(example, nli_pipe, model):
    eps = 1e-9
    with Timer() as timer:
        # Bidirectional Entailment Clustering
        washed_sampled_answer = example['washed_sampled_answer']
        meanings = [[washed_sampled_answer[0]]]
        seqs = washed_sampled_answer[1:]
        for s in seqs:
            in_existing_meaning = False
            for c in meanings:
                s_c = c[0]
                tmp = "[CLS] {s1} [SEP] {s2} [CLS]"
                res = nli_pipe([tmp.format(s1=s, s2=s_c), tmp.format(s1=s_c, s2=s)])
                if res[0]['label'] == 'ENTAILMENT' and res[1]['label'] == 'ENTAILMENT':
                    c.append(s)
                    in_existing_meaning = True
                    break
            if not in_existing_meaning:
                meanings.append([s])
        # Calculate Semantic Entropy
        pcs = []
        for c in meanings:
            pc = eps
            for s in c:
                idx = example['washed_sampled_answer'].index(s)
                answer_prob = example['sampled_answer_prob'][idx]
                ps = np.prod(answer_prob)
                pc += ps
            pcs.append(pc)
        example['u_score_se'] = -np.sum(np.log(pcs) * pcs)
    example['time_se'] = timer.t
    return example


def get_uncertainty_score_sar_all(example, sar_bert, T, model):
    if example['washed_answer'] == "":
        example['u_score_token_sar'] = 1
        example['u_score_sent_sar'] = 1
        example['u_score_sar'] = 1
        example['time_token_sar'] = 0
        example['time_sent_sar'] = 0
        example['time_sar'] = 0
        return example

    def get_token_sar(input_ids, answer_ids, answer_prob):
        output_ids = input_ids + answer_ids
        orig_embedding = sar_bert.encode(model.to_string(output_ids), convert_to_tensor=True).to(sar_bert.device)
        neg_logp = -torch.log(torch.tensor(answer_prob))
        new_input_strings = []
        for j in range(len(input_ids), len(input_ids + answer_ids)):
            new_input_ids = output_ids[:j] + output_ids[j + 1:]
            new_input_string = model.to_string(new_input_ids)
            new_input_strings.append(new_input_string)
        # print("new_input_strings",new_input_strings)
        # print(len(new_input_strings))
        if not new_input_strings:
            return 1
        new_embeddings = sar_bert.encode(new_input_strings, convert_to_tensor=True).to(sar_bert.device)
        sim = st_util.dot_score(F.normalize(orig_embedding, dim=-1), F.normalize(new_embeddings, dim=-1))[0].cpu()
        # sim = (sim + 1) / 2
        # print(sim)
        rt = 1 - sim
        # print(f"rt:{rt.shape}")
        # print(f"neg_logp:{neg_logp.shape}")
        # print(f"answer_ids:{answer_ids}")
        # print(f"answer_prob:{answer_prob}")
        rt = rt / rt.sum()
        token_sar = einsum('s, s ->', neg_logp, rt).item()
        return token_sar

    def get_sent_sar(gen_prob, sim):
        rs = sim * gen_prob.repeat(len(sim), 1)
        rs[torch.arange(len(rs)), torch.arange(len(rs))] = 0
        rs = rs.sum(dim=-1)
        es = -torch.log(gen_prob.squeeze() + rs / T)
        score = es.mean().item()
        return score

    with Timer() as timer:
        token_sar = get_token_sar(example['input_ids'], example['washed_answer_ids'], example['answer_prob'])
        example['u_score_token_sar'] = token_sar
    example['time_token_sar'] = timer.t

    if not example.get('sampled_answer_prob'):
        example = get_sampled_answer_prob(example, model)

    with Timer() as timer:
        embeddings = sar_bert.encode(example['washed_sampled_answer'], convert_to_tensor=True)
        sim = cos_sim(embeddings, embeddings).cpu()
        sim = (sim + 1) / 2
    example['time_sent_sar'] = timer.t
    example['time_sar'] = timer.t

    with Timer() as timer:
        gen_prob = torch.tensor(list(map(np.prod, example['sampled_answer_prob']))).unsqueeze(0)
        # print('gen_prob: ', gen_prob)
        sent_sar = get_sent_sar(gen_prob, sim)
        # print('sent_sar: ', sent_sar)
        example['u_score_sent_sar'] = sent_sar
    example['time_sent_sar'] += timer.t

    with Timer() as timer:
        gen_prob = []
        for answer_ids, answer_prob in zip(example['washed_sampled_answer_ids'], example['sampled_answer_prob']):
            gen_prob.append(math.exp(-get_token_sar(example['input_ids'], answer_ids, answer_prob)))
        gen_prob = torch.tensor(gen_prob).unsqueeze(0)
        # print('gen_prob: ', gen_prob)
        sar = get_sent_sar(gen_prob, sim)
        # print('sar: ', sar)
        example['u_score_sar'] = sar
    example['time_sar'] += timer.t
    return example


def get_uncertainty_score_len(example):
    example['u_score_len'] = example['num_answer_tokens']
    return example


class VcModel(nn.Module):
    def __init__(self, model: HookedTransformer, hooked_layers: List[int], hooked_act_name: str, head_type: str, pool_type: str, mlp_hidden_size=None):
        super().__init__()
        self.model = model
        self.hooked_layers = hooked_layers
        self.hooked_act_name = hooked_act_name
        self.head_type = head_type
        self.pool_type = pool_type
        
        
        self.hooked_module_names = [utils.get_act_name(self.hooked_act_name, l) for l in sorted(self.hooked_layers)]
        self.vc_module_names = [name.replace(".", "#") for name in self.hooked_module_names]
        self.mlp_hidden_size = self.model.cfg.d_model //16 if mlp_hidden_size is None else mlp_hidden_size
        
        def head_func(head_type):
            if head_type == "linear":
                return nn.Linear(self.model.cfg.d_model, 1)
            elif head_type == "mlp":
                return nn.Sequential(
                    nn.Linear(self.model.cfg.d_model, self.model.cfg.d_model),
                    nn.ReLU(),
                    nn.Linear(self.model.cfg.d_model, 1)
                )
            elif head_type == "mlp_small":
                return nn.Sequential(
                    nn.Linear(self.model.cfg.d_model, self.mlp_hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.mlp_hidden_size, 1)
                )
            elif head_type == "mlp_unembed":
                return nn.Sequential(
                    self.model.ln_final,
                    self.model.unembed,
                    nn.Linear(self.model.cfg.d_model, self.model.cfg.d_model),
                    nn.ReLU(),
                    nn.Linear(self.model.cfg.d_model, 1)
                )
            else:
                raise ValueError(f"head_type {head_type} not supported")
        
        self.vc = nn.ModuleDict({
                vc_module_name: head_func(head_type)
                for vc_module_name in self.vc_module_names
            })
        
        if "attn" in self.pool_type:
            self.vc_q = nn.ModuleDict({
                vc_module_name: head_func(head_type)
                for vc_module_name in self.vc_module_names
            })

    def forward_with_uncertainty_hook(self, input_ids, answer_ids):
        bsz = len(input_ids)
        u_score_dict = {}
        layer_batch_pos_scores = []
        layer_batch_pos_weights = []
        
        padded_output_ids, mask = _get_batch_padded_output_ids(input_ids, answer_ids, self.model.tokenizer.pad_token_id, 'right')
        mask = mask.to(self.model.cfg.device)
        
        def layer_score_hook(resid: Float[torch.Tensor, 'b p d'], hook: HookPoint):
            vc_l = self.vc[hook.name.replace(".", "#")]
            batch_pos_scores = F.sigmoid(vc_l(resid).squeeze(-1)) # [b p d] -> [b p]
            layer_batch_pos_scores.append(batch_pos_scores)
            return resid
        
        def layer_score_attn_hook(resid: Float[torch.Tensor, 'b p d'], hook: HookPoint):
            vc_l = self.vc[hook.name.replace(".", "#")]
            vc_q_l = self.vc_q[hook.name.replace(".", "#")]
            batch_pos_scores = F.sigmoid(vc_l(resid).squeeze(-1)) # [b p d] -> [b p]
            # batch_pos_weights = vc_q_l(F.normalize(resid, p=2, dim=-1)).squeeze(-1) # [b p d] -> [b p]
            batch_pos_weights = vc_q_l(resid).squeeze(-1) # [b p d] -> [b p]
            batch_pos_weights = torch.where(mask==1, batch_pos_weights, torch.finfo(resid.dtype).min)
            layer_batch_pos_weights.append(batch_pos_weights)
            layer_batch_pos_scores.append(batch_pos_scores)
            return resid
        
        if "attn" in self.pool_type:
            fwd_hooks = [(lambda x: x in self.hooked_module_names, layer_score_attn_hook)]
        else:
            fwd_hooks = [(lambda x: x in self.hooked_module_names, layer_score_hook)]
            
        model_output = self.model.run_with_hooks(padded_output_ids, fwd_hooks=fwd_hooks, prepend_bos=False, padding_side='right')
        
        batch_layer_pos_scores = einops.rearrange(layer_batch_pos_scores, 'l b p -> b l p')
        
        if "attn" in self.pool_type:
            if self.pool_type == "attn_all":
                batch_layer_pos_weights = einops.rearrange(layer_batch_pos_weights, 'l b p -> b (l p)')
                batch_layer_pos_weights = F.softmax(batch_layer_pos_weights, dim=-1)
                batch_layer_pos_weights = einops.rearrange(batch_layer_pos_weights, 'b (l p) -> b l p', l=len(self.hooked_layers))
            if self.pool_type == "attn_all_no_first":
                batch_layer_pos_weights = einops.rearrange(layer_batch_pos_weights, 'l b p -> b l p')
                batch_layer_pos_weights[:,:,0] = torch.finfo(batch_layer_pos_weights.dtype).min
                batch_layer_pos_weights = einops.rearrange(batch_layer_pos_weights, 'b l p -> b (l p)')
                batch_layer_pos_weights = F.softmax(batch_layer_pos_weights, dim=-1)
                batch_layer_pos_weights = einops.rearrange(batch_layer_pos_weights, 'b (l p) -> b l p', l=len(self.hooked_layers))
            if self.pool_type == "attn_token_all":
                batch_layer_pos_weights = einops.rearrange(layer_batch_pos_weights, 'l b p -> b l p')
                batch_layer_pos_weights = F.softmax(batch_layer_pos_weights, dim=-1) / len(self.hooked_layers)
            if self.pool_type == "attn_token_all_no_first":
                batch_layer_pos_weights = einops.rearrange(layer_batch_pos_weights, 'l b p -> b l p')
                batch_layer_pos_weights[:,:,0] = torch.finfo(batch_layer_pos_weights.dtype).min
                batch_layer_pos_weights = F.softmax(batch_layer_pos_weights, dim=-1) / len(self.hooked_layers)
            if self.pool_type == "attn_token_ans":
                batch_layer_pos_weights = einops.rearrange(layer_batch_pos_weights, 'l b p -> b l p')
                for i in range(bsz):
                    batch_layer_pos_weights[i,:,len(input_ids[i]):] = torch.finfo(batch_layer_pos_weights.dtype).min
                batch_layer_pos_weights = F.softmax(batch_layer_pos_weights, dim=-1) / len(self.hooked_layers)
            if self.pool_type == "attn_token_inp":
                batch_layer_pos_weights = einops.rearrange(layer_batch_pos_weights, 'l b p -> b l p')
                for i in range(bsz):
                    batch_layer_pos_weights[i,:,:len(input_ids[i])] = torch.finfo(batch_layer_pos_weights.dtype).min
                batch_layer_pos_weights = F.softmax(batch_layer_pos_weights, dim=-1) / len(self.hooked_layers)
            batch_layer_pos_scores = batch_layer_pos_weights * batch_layer_pos_scores
            
        batch_layer_ans_scores = []
        batch_layer_inp_scores = []
        batch_layer_all_scores = []
        batch_layer_all_weights = []
        
        for i in range(bsz):
            lps = batch_layer_pos_scores[i]
            inp_len = len(input_ids[i])
            ans_len = len(answer_ids[i])
            batch_layer_inp_scores.append(lps[:, :inp_len])
            batch_layer_ans_scores.append(lps[:, inp_len:inp_len + ans_len])
            batch_layer_all_scores.append(lps[:, :inp_len + ans_len])
            if "attn" in self.pool_type:
                batch_layer_all_weights.append(batch_layer_pos_weights[i][:, :inp_len + ans_len])
        
        u_score_dict['u_score_all'] = batch_layer_all_scores
        u_score_dict['u_weight'] = batch_layer_all_weights
        
        # Pooling
        default_zero = torch.zeros_like(batch_layer_pos_scores[0][0][0])
        if self.pool_type == "mean_ans":
            batch_scores = list(map(lambda x: x.mean() if x.numel() != 0 else default_zero, batch_layer_ans_scores))
        elif self.pool_type == "mean_inp":
            batch_scores = list(map(lambda x: x.mean() if x.numel() != 0 else default_zero, batch_layer_inp_scores))
        elif self.pool_type == "mean_all":
            batch_scores = list(map(lambda x: x.mean() if x.numel() != 0 else default_zero, batch_layer_all_scores))
        elif self.pool_type == "last_ans":
            batch_scores = list(map(lambda x: x[:,-1].mean() if x.numel() != 0 else default_zero, batch_layer_ans_scores))
        elif self.pool_type == "last_inp":
            batch_scores = list(map(lambda x: x[:,-1].mean() if x.numel() != 0 else default_zero, batch_layer_inp_scores))
        elif 'attn' in self.pool_type:
            batch_scores = list(map(lambda x: x.sum() if x.numel() != 0 else default_zero, batch_layer_all_scores))
        else:
            raise ValueError(f"pool_type {self.pool_type} not supported")
        
        batch_scores = torch.stack(batch_scores)
        # batch_scores[batch_scores.isnan()] = 0
        u_score_dict['u_score'] = batch_scores
        
        return model_output, u_score_dict
    
    def save_to_disk(self, save_path):
        config = dict(
            hooked_layers=self.hooked_layers,
            hooked_act_name=self.hooked_act_name,
            head_type=self.head_type,
            pool_type=self.pool_type
        )
        model = dict(
            vc=self.vc.state_dict()
        )
        if hasattr(self, 'vc_q'):
            model['vc_q'] = self.vc_q.state_dict()
        model_and_config = dict(config=config, model=model)
        torch.save(model_and_config, save_path)
    
    @classmethod
    def load_from_disk(cls, model, save_path):
        model_and_config = torch.load(save_path)
        config = model_and_config['config']
        vc_model = cls(model, **config)
        vc_model.vc.load_state_dict(model_and_config['model']['vc'])
        vc_model.vc.to(vc_model.model.W_E.device).to(vc_model.model.W_E.dtype)
        if "attn" in vc_model.pool_type:
            vc_model.vc_q.load_state_dict(model_and_config['model']['vc_q'])
            vc_model.vc_q.to(vc_model.model.W_E.device).to(vc_model.model.W_E.dtype)
        return vc_model


def get_uncertainty_score_ours_all(examples, vc_model):
    bsz = len(examples['input'])
    with Timer() as timer:
        model_output, u_score_dict = vc_model.forward_with_uncertainty_hook(examples['input_ids'], examples['washed_answer_ids'])
        examples[f'u_score_ours_{vc_model.head_type}_{vc_model.pool_type}'] = (1 - u_score_dict['u_score']).float().cpu().numpy().tolist()
    examples['time_ours'] = [timer.t / bsz for i in range(bsz)]
    return examples

# Evaluation: AUROC with Correctness Metric
def get_auroc(val_dst, u_metric, c_metric, c_th):
    c_metrics = val_dst[c_metric]
    label = [0 if c > c_th else 1 for c in c_metrics]
    if all(l == 0 for l in label) or all(l == 1 for l in label):
        return 0.5
    u_score = val_dst[u_metric]
    u_score = [u if not math.isnan(u) else 0 for u in u_score]
    auroc = roc_auc_score(label, u_score)
    return auroc


def plot_th_curve(test_dst, u_metrics, c_metric, nbins=20):
    fig = go.Figure()
    th_range = [i / nbins for i in range(1, nbins)]
    accs = []
    c_metrics = test_dst[c_metric]

    for th in th_range:
        acc = 0
        for c in c_metrics:
            if c > th:
                acc += 1
        acc = acc / len(c_metrics)
        accs.append(acc)

    fig.add_trace(go.Scatter(x=th_range, y=accs, mode='lines+markers+text', name=f"acc", text=[f"{a:.4f}" for a in accs], textposition="top center"))

    for u_metric in u_metrics:
        aurocs = []
        for acc, th in zip(accs, th_range):
            aurocs.append(get_auroc(test_dst, u_metric, c_metric, th))
        fig.add_trace(go.Scatter(x=th_range, y=aurocs, mode='lines+markers+text', name=f"{u_metric}", text=[f"{a:.4f}" for a in aurocs], textposition="top center"))
    fig.update_layout(title=f"AUROC/{c_metric}-Threshold Curve", xaxis_title=f"{c_metric}-Threshold", yaxis_title="AUROC", width=2000, height=1000)

    return fig