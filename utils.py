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
import math
import fire
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


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
def wash(text, tokenizer, first_sentence_only):
    for sp_tok in tokenizer.special_tokens_map.values():
        text = text.replace(sp_tok, "")

    first_string_before_question = text
    spliters = ['question:', 'context:']
    for spliter in spliters:
        if spliter in text.lower():
            first_string_before_question = text.lower().split(spliter)[0]
            break
    text = text[:len(first_string_before_question)]

    if first_sentence_only:
        text = text.split('.')[0]

    text = text.strip()

    return text


def wash_answer(example, tokenizer, first_sentence_only):
    example['washed_answer'] = wash(example['answer'], tokenizer, first_sentence_only)
    example['washed_output'] = example['input'] + example['washed_answer']
    if example.get("sampled_answer"):
        example['washed_sampled_answer'] = [wash(ans, tokenizer, first_sentence_only) for ans in example['sampled_answer']]
        example['washed_sampled_output'] = [example['input'] + ans for ans in example['washed_sampled_answer']]
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
    cosine_scores = torch.diag(cos_sim(embeddings1, embeddings2))
    cosine_scores = (cosine_scores + 1) / 2
    batch_sentsim = cosine_scores.tolist()
    examples['sentsim'] = batch_sentsim
    return examples

def get_include(example):
    wrong_answers = []
    if example.get('options', []):
        wrong_answers = [example['options']]
    include = 0
    if example['gt'] in example['washed_answer']:
        include = 1
        for wa in wrong_answers:
            if wa in example['washed_answer']:
                include = 0
                break
    example['include'] = include
    return example

def get_num_tokens(examples, tokenizer):
    batch_num_input_tokens = list(map(len, tokenizer(examples['input'])['input_ids']))
    batch_num_output_tokens = list(map(len, tokenizer(examples['washed_output'])['input_ids']))
    batch_num_answer_tokens = [num_output_tokens - num_input_tokens for num_input_tokens, num_output_tokens in zip(batch_num_input_tokens, batch_num_output_tokens)]
    batch_answer_idxs = [list(range(-num_answer_tokens - 1, 0)) for num_answer_tokens in batch_num_answer_tokens]
    examples['num_input_tokens'] = batch_num_input_tokens
    examples['num_output_tokens'] = batch_num_output_tokens
    examples['num_answer_tokens'] = batch_num_answer_tokens
    examples['answer_idxs'] = batch_answer_idxs
    return examples


def _get_answer_prob(inp, out, prob, model):
    num_input_tokens = len(model.to_str_tokens(inp))
    output_tokens = model.to_tokens(out, move_to_device=False)[0].tolist()
    if len(output_tokens) == num_input_tokens:
        return []
    answer_tokens = output_tokens[num_input_tokens:]
    answer_prob = prob[num_input_tokens - 1:-1, :]
    answer_prob = answer_prob[range(len(answer_tokens)), answer_tokens]
    answer_prob = answer_prob.tolist()
    return answer_prob


def get_answer_prob(examples, model):
    batch_answer_prob = []
    bsz = len(examples['input'])
    with Timer() as timer:
        batch_logits = model(examples['washed_output'], padding_side='right')

    batch_prob = F.softmax(batch_logits, dim=-1)  # prob: (bsz pos vocab)

    for i in range(len(examples['washed_output'])):
        answer_prob = _get_answer_prob(examples['input'][i], examples['washed_output'][i], batch_prob[i], model)
        batch_answer_prob.append(answer_prob)

    examples['answer_prob'] = batch_answer_prob
    examples['time_fwd'] = [timer.t / bsz for i in range(bsz)]
    return examples


def get_sampled_answer_prob(example, model):
    batch_answer_prob = []
    washed_sampled_output = example['washed_sampled_output']
    washed_sampled_output_unique = list(set(washed_sampled_output))
    batch_prob = F.softmax(model(washed_sampled_output_unique, padding_side='right'), dim=-1)  # logits: (bsz pos vocab)

    for i in range(len(example['washed_sampled_output'])):
        inp = example['input']
        out = example['washed_sampled_output'][i]
        prob = batch_prob[washed_sampled_output_unique.index(out)]
        answer_prob = _get_answer_prob(inp, out, prob, model)
        batch_answer_prob.append(answer_prob)

    example['sampled_answer_prob'] = batch_answer_prob
    return example


# Uncertainty Estimation Baselines
def get_uncertainty_score_token_pe_all(examples, model):
    with Timer() as timer:
        if not examples.get('answer_prob'):
            examples = get_answer_prob(examples, model)
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
        sampled_outputs = example['washed_sampled_answer']
        rouge = Rouge()
        hyps = []
        refs = []
        for i in range(len(sampled_outputs)):
            for j in range(i + 1, len(sampled_outputs)):
                hyp = sampled_outputs[i]
                ref = sampled_outputs[j]
                if hyp == "" or hyp == '.':
                    hyp = "-"
                if ref == "" or ref == '.':
                    ref = "-"
                hyps.append(hyp)
                refs.append(ref)
        scores = rouge.get_scores(hyps, refs, avg=True)
        example['u_score_ls'] = scores['rouge-l']['f']
    example['time_ls'] = timer.t
    return example


def get_uncertainty_score_se(example, nli_pipe, model):
    eps = 1e-9
    # Sample Answers
    washed_sampled_answer = example['washed_sampled_answer']
    if not example.get('sampled_answer_prob'):
        example = get_sampled_answer_prob(example, model)
    with Timer() as timer:
        # Bidirectional Entailment Clustering
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
    if not example.get('answer_prob'):
        example = get_answer_prob(example, model)

    if example['washed_answer'] == "":
        example['u_score_token_sar'] = 0
        example['u_score_sent_sar'] = 0
        example['u_score_sar'] = 0
        return example

    def get_token_sar(inp, out, answer_prob):
        num_input_tokens = len(model.to_str_tokens(inp))
        num_output_tokens = len(model.to_str_tokens(out))
        orig_embedding = sar_bert.encode(out, convert_to_tensor=True)
        neg_logp = -torch.log(torch.tensor(answer_prob))
        input_tokens = model.to_tokens(out, move_to_device=False)[0].tolist()
        start, end = num_input_tokens, num_output_tokens
        new_input_strings = []
        for j in range(start, end):
            new_input_tokens = input_tokens[:j] + input_tokens[j + 1:]
            new_input_string = model.to_string(new_input_tokens)
            new_input_strings.append(new_input_string)
        if not new_input_strings:
            return 0
        new_embeddings = sar_bert.encode(new_input_strings, convert_to_tensor=True)
        orig_embedding = orig_embedding.to(sar_bert.device)
        new_embeddings = new_embeddings.to(sar_bert.device)
        sim = cos_sim(orig_embedding, new_embeddings)[0].cpu()
        sim = (sim + 1) / 2
        rt = 1 - sim
        rt = rt / rt.sum()
        token_sar = einsum('s, s ->', neg_logp, rt).item()
        return token_sar

    with Timer() as timer:
        token_sar = get_token_sar(example['input'], example['washed_output'], example['answer_prob'])
        example['u_score_token_sar'] = token_sar
    example['time_token_sar'] = timer.t

    if not example.get('sampled_answer_prob'):
        example = get_sampled_answer_prob(example, model)

    with Timer() as timer:
        embeddings = sar_bert.encode(example['washed_sampled_answer'], convert_to_tensor=True)
        cosine_scores = cos_sim(embeddings, embeddings).cpu()
        sim = (cosine_scores + 1) / 2
    example['time_sent_sar'] = timer.t
    example['time_sar'] = timer.t

    with Timer() as timer:
        gen_prob = torch.tensor(list(map(np.prod, example['sampled_answer_prob']))).unsqueeze(0)
        rs = sim * gen_prob.repeat(len(sim), 1)
        rs[torch.arange(len(rs)), torch.arange(len(rs))] = 0
        rs = rs.sum(dim=-1)
        es = -torch.log(gen_prob.squeeze() + rs / T)
        sent_sar = es.mean().item()
        example['u_score_sent_sar'] = sent_sar
    example['time_sent_sar'] += timer.t

    with Timer() as timer:
        gen_prob = []
        for out, answer_prob in zip(example['washed_sampled_output'], example['sampled_answer_prob']):
            gen_prob.append(math.exp(-get_token_sar(example['input'], out, answer_prob)))
        gen_prob = torch.tensor(gen_prob).unsqueeze(0)
        rs = sim * gen_prob.repeat(len(sim), 1)
        rs[torch.arange(len(rs)), torch.arange(len(rs))] = 0
        rs = rs.sum(dim=-1)
        es = -torch.log(gen_prob.squeeze() + rs / T)
        sar = es.mean().item()
        example['u_score_sar'] = sar
    example['time_sar'] += timer.t
    return example


def get_uncertainty_score_len(example):
    example['u_score_len'] = example['num_answer_tokens']
    return example


def get_uncertainty_score_ours_all(examples, v_c, score_func, label_type, model):
    bsz = len(examples['input'])
    with Timer() as timer:
        full_act_names = [k.replace('#', '.') for k in v_c.keys()]
        layer_batch_scores = []

        def score_hook(resid: Float[torch.Tensor, 'b p d'], hook: HookPoint):
            v_c_l = v_c[hook.name.replace(".", "#")]
            r = resid[:, -max(examples['num_answer_tokens']) - 2:, :]
            batch_all_scores = v_c_l(r)  # [b p d] -> [b p 1]
            batch_all_scores = batch_all_scores.squeeze()
            batch_scores = []
            for scores, idxs in zip(batch_all_scores, examples['answer_idxs']):
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

        fwd_hooks = [(lambda x: x in full_act_names, score_hook)]
        out = model.run_with_hooks(examples['washed_output'], fwd_hooks=fwd_hooks)

        batch_scores = einops.reduce(layer_batch_scores, 'l b -> b', 'mean')
        examples[f'u_score_ours_{score_func}_{label_type}'] = batch_scores.tolist()

    examples[f'time_ours_{score_func}_{label_type}'] = [timer.t / bsz - examples['time_fwd'][i] for i in range(bsz)]
    return examples


# Evaluation: AUROC with Correctness Metric
def get_auroc(val_dst, u_metric, c_metric, c_th):
    c_metrics = val_dst[c_metric]
    label = [1 if c > c_th else 0 for c in c_metrics]
    u_score = val_dst[u_metric]
    auroc = roc_auc_score(label, u_score)
    auroc = auroc if auroc > 0.5 else 1 - auroc
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
            if acc == 0 or acc == 1:
                continue
            aurocs.append(get_auroc(test_dst, u_metric, c_metric, th))
        fig.add_trace(go.Scatter(x=th_range, y=aurocs, mode='lines+markers+text', name=f"{u_metric}", text=[f"{a:.4f}" for a in aurocs], textposition="top center"))
    fig.update_layout(title=f"AUROC/{c_metric}-Threshold Curve", xaxis_title=f"{c_metric}-Threshold", yaxis_title="AUROC", width=2000, height=1000)

    return fig


def rescale_uscore(example, u_metric, mean, std):
    old_score = example[u_metric]
    new_score = ((old_score - mean) / std + 1) / 2
    example[u_metric] = new_score
    return example
