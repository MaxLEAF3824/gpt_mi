import math
import os

os.environ['HF_DATASETS_OFFLINE'] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import transformer_lens
import datasets
import pandas as pd
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
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
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
    cosine_scores = torch.diag(cos_sim(embeddings1, embeddings2))
    # cosine_scores = (cosine_scores + 1) / 2
    batch_sentsim = cosine_scores.tolist()
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
    max_len = max(map(len, batch_output_ids))
    if padding_side == "right":
        padded_output_ids = torch.cat([F.pad(output_ids, (0, max_len - len(output_ids)), value=pad_token_id).unsqueeze(0) for output_ids in batch_output_ids], dim=0)
    elif padding_side == "left":
        padded_output_ids = torch.cat([F.pad(output_ids, (max_len - len(output_ids), 0), value=pad_token_id).unsqueeze(0) for output_ids in batch_output_ids], dim=0)
    else:
        raise ValueError(f"padding_side {padding_side} not supported")

    return padded_output_ids


def get_answer_prob(examples, model):
    bsz = len(examples['input'])
    batch_answer_prob = []
    padded_output_ids = _get_batch_padded_output_ids(examples['input_ids'], examples['washed_answer_ids'], model.tokenizer.pad_token_id, 'right')
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
    padded_output_ids = _get_batch_padded_output_ids([example['input_ids']] * len(uni_ids), list(map(list, uni_ids)), model.tokenizer.pad_token_id, 'right')
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
        example['u_score_ls'] = scores['rouge-l']['f']
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
        example['u_score_token_sar'] = 0
        example['u_score_sent_sar'] = 0
        example['u_score_sar'] = 0
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
            return 0
        new_embeddings = sar_bert.encode(new_input_strings, convert_to_tensor=True).to(sar_bert.device)
        sim = cos_sim(orig_embedding, new_embeddings)[0].cpu()
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
    example['time_sent_sar'] = timer.t
    example['time_sar'] = timer.t

    with Timer() as timer:
        gen_prob = torch.tensor(list(map(np.prod, example['sampled_answer_prob']))).unsqueeze(0)
        example['u_score_sent_sar'] = get_sent_sar(gen_prob, sim)
    example['time_sent_sar'] += timer.t

    with Timer() as timer:
        gen_prob = []
        for answer_ids, answer_prob in zip(example['washed_sampled_answer_ids'], example['sampled_answer_prob']):
            gen_prob.append(math.exp(-get_token_sar(example['input_ids'], answer_ids, answer_prob)))
        gen_prob = torch.tensor(gen_prob).unsqueeze(0)
        example['u_score_sar'] = get_sent_sar(gen_prob, sim)
    example['time_sar'] += timer.t
    return example


def get_uncertainty_score_len(example):
    example['u_score_len'] = example['num_answer_tokens']
    return example


def ours_forward_func(examples, v_c, score_func, model):
    layer_batch_scores = []

    def score_hook(resid: Float[torch.Tensor, 'b p d'], hook: HookPoint):
        v_c_l = v_c[hook.name.replace(".", "#")]
        r = resid[:, -max(examples['num_answer_tokens']) - 2:, :]
        batch_all_scores = v_c_l(r)  # [b p d] -> [b p 1]
        batch_all_scores = batch_all_scores.squeeze()  # [b p 1] -> [b p]
        batch_scores = []
        for scores, idxs in zip(batch_all_scores, examples['answer_idxs']):
            if not idxs:
                s = torch.zeros_like(scores[0])
                batch_scores.append(s)
                continue
            answer_scores = scores[idxs]
            if score_func == "sum":
                s = answer_scores.sum()
            elif score_func == "mean":
                s = answer_scores.mean()
            elif score_func == "last":
                s = answer_scores[-1]
            elif score_func == "max":
                s = answer_scores.max()
            else:
                raise ValueError(f"score_func {score_func} not supported")

            batch_scores.append(s)
        batch_scores = torch.stack(batch_scores)
        layer_batch_scores.append(batch_scores)
        return resid

    full_act_names = [k.replace('#', '.') for k in v_c.keys()]
    fwd_hooks = [(lambda x: x in full_act_names, score_hook)]
    padded_output_ids = _get_batch_padded_output_ids(examples['input_ids'], examples['washed_answer_ids'], model.tokenizer.pad_token_id, 'left')
    out = model.run_with_hooks(padded_output_ids, fwd_hooks=fwd_hooks, prepend_bos=False, padding_side='left')

    batch_scores = einops.reduce(layer_batch_scores, 'l b -> b', 'mean')
    return batch_scores


def get_uncertainty_score_ours_all(examples, v_c, score_func, label_type, label_name, model):
    bsz = len(examples['input'])
    with Timer() as timer:
        batch_scores = ours_forward_func(examples, v_c, score_func, model)
        examples[f'u_score_ours_{score_func}_{label_type}_{label_name}'] = batch_scores.tolist()

    examples[f'time_ours_{score_func}_{label_type}_{label_name}'] = [timer.t / bsz for i in range(bsz)]
    return examples


# Evaluation: AUROC with Correctness Metric
def get_auroc(val_dst, u_metric, c_metric, c_th):
    c_metrics = val_dst[c_metric]
    label = [1 if c > c_th else 0 for c in c_metrics]
    u_score = val_dst[u_metric]
    auroc = roc_auc_score(label, u_score)
    # auroc = auroc if auroc > 0.5 else 1 - auroc
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
            if acc == 0 or acc == 1.:
                aurocs.append(0.5)
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


# Our Method OLD
def compute_certainty_vector_mean(train_dst, model, dst_name, layers, act_name, batch_size=8, ):
    def get_paired_dst_sciq(train_dst):
        tmp_pos = "Question:{q} Options:{o} The correct answer is:"
        tmp_neg = "Question:{q} Options:{o} The incorrect answer is:"

        # sciq_train_dst = sciq_train_dst.filter(lambda x: x['rougel'] > 0.5)

        def get_pos_example(example):
            example['input'] = tmp_pos.format(q=example['question'], o=", ".join(example['options']))
            example['washed_output'] = f"{example['input']}{example['gt']}"
            return example

        def get_neg_example(example, idx):
            example['input'] = tmp_neg.format(q=example['question'], o=", ".join(example['options']))
            wrong_options = [opt for opt in example['options'] if opt != example['gt']]
            if wrong_options:
                random.seed(42 + idx)
                wrong_answer = random.choice(wrong_options)
            else:
                wrong_answer = "wrong answer"
            example['washed_output'] = f"{example['input']}{wrong_answer}"
            return example

        dst_pos = train_dst.map(get_pos_example, new_fingerprint=str(time()))
        dst_neg = train_dst.map(get_neg_example, with_indices=True, new_fingerprint=str(time()))
        return dst_pos, dst_neg

    def get_paired_dst_coqa(train_dst):
        def get_pos_example(example):
            example['washed_output'] = f"{example['input']}The correct answer is {example['gt']}"
            return example

        def get_neg_example(example, idx):
            wrong_options = [opt for opt in example['answers']['input_text'] if opt != example['gt']]
            if wrong_options:
                random.seed(42 + idx)
                wrong_answer = random.choice(wrong_options)
            else:
                wrong_answer = "wrong answer"
            example['washed_output'] = f"{example['input']}The wrong answer is {wrong_answer}"
            return example

        dst_pos = train_dst.map(get_pos_example, new_fingerprint=str(time()))
        dst_neg = train_dst.map(get_neg_example, with_indices=True, new_fingerprint=str(time()))
        return dst_pos, dst_neg

    def get_paired_dst_triviaqa(train_dst):
        def get_pos_example(example):
            example['washed_output'] = f"{example['input']}The correct answer is {example['gt']}"
            return example

        def get_neg_example(example, idx):
            next_idx = idx + 1 if idx + 1 < len(train_dst) else 0
            wrong_answer = train_dst[next_idx]['gt']
            example['washed_output'] = f"{example['input']}The wrong answer is {wrong_answer}"
            return example

        dst_pos = train_dst.map(get_pos_example, new_fingerprint=str(time()))
        dst_neg = train_dst.map(get_neg_example, with_indices=True, new_fingerprint=str(time()))
        return dst_pos, dst_neg

    def get_paired_dst_medmcqa(train_dst):
        def get_pos_example(example):
            example['washed_output'] = f"{example['input']}The correct answer is {example['gt']}"
            return example

        def get_neg_example(example, idx):
            wrong_options = [opt for opt in example['options'] if opt != example['gt']]
            if wrong_options:
                random.seed(42 + idx)
                wrong_answer = random.choice(wrong_options)
            else:
                wrong_answer = "wrong answer"
            example['washed_output'] = f"{example['input']}The wrong answer is {wrong_answer}"
            return example

        dst_pos = train_dst.map(get_pos_example, new_fingerprint=str(time()))
        dst_neg = train_dst.map(get_neg_example, with_indices=True, new_fingerprint=str(time()))
        return dst_pos, dst_neg

    func_map = {
        'allenai/sciq': get_paired_dst_sciq,
        'stanfordnlp/coqa': get_paired_dst_coqa,
        'lucadiliello/triviaqa': get_paired_dst_triviaqa,
        'openlifescienceai/medmcqa': get_paired_dst_medmcqa,
        'GBaker/MedQA-USMLE-4-options': get_paired_dst_medmcqa
    }
    func = func_map[dst_name]
    dst_pos, dst_neg = func(train_dst)

    data_pos = dst_pos['washed_output']
    data_neg = dst_neg['washed_output']
    data_size = len(data_pos)
    full_act_names = [utils.get_act_name(act_name, l) for l in sorted(layers)]
    v_c = torch.zeros((len(layers), 1, model.cfg.d_model)).cuda()

    for i in tqdm(range(0, data_size, batch_size)):
        batch_pos = data_pos[i:i + batch_size]
        batch_neg = data_neg[i:i + batch_size]

        _, cache_pos = model.run_with_cache(batch_pos, names_filter=lambda x: x in full_act_names, padding_side='left')  # logits: (bsz pos vocab) cache: dict
        _, cache_neg = model.run_with_cache(batch_neg, names_filter=lambda x: x in full_act_names, padding_side='left')  # logits: (bsz pos vocab) cache: dict

        cache_pos = einops.rearrange([cache_pos[name] for name in full_act_names], 'l b p d -> b l p d')
        cache_neg = einops.rearrange([cache_neg[name] for name in full_act_names], 'l b p d -> b l p d')

        cache_pos = cache_pos[:, :, [-1], :]
        cache_neg = cache_neg[:, :, [-1], :]

        v_c += (cache_pos.sum(dim=0) - cache_neg.sum(dim=0))

    v_c /= data_size

    v_c = v_c.cpu().float()
    v_c = F.normalize(v_c, p=2, dim=-1)
    return v_c


# clean_exp exp
def clean_exp(dst, model, v_c, layers, act_name):
    fig = go.Figure()
    c_scores = []
    w_scores = []
    labels = []
    u_scores = []
    u_scores_z = []
    all_pe_u_scores = []
    all_ln_pe_u_scores = []

    def batch_get_result(examples):
        all_outputs = []
        all_num_answer_tokens = []
        all_num_input_tokens = list(map(len, model.to_str_tokens(examples['input'])))
        bsz = len(examples['input'])

        for i in range(bsz):
            example = {k: examples[k][i] for k in examples.keys()}
            if example.get("options"):
                wrong_options = [opt for opt in example['options']]
                for opt in wrong_options:
                    if opt == example['gt']:
                        wrong_options.remove(opt)
                        break
            elif example.get("answers"):
                wrong_options = [opt for opt in example['answers']['input_text']]
                for opt in wrong_options:
                    if opt == example['gt']:
                        wrong_options.remove(opt)
                        break
                wrong_options = wrong_options[:3]
            else:
                wrong_options = ['wrong answer', 'bad answer', 'incorrect answer']
            correct_output = example['input'] + example['gt']
            wrong_outputs = [example['input'] + opt for opt in wrong_options]
            all_outputs.extend([correct_output] + wrong_outputs)
            num_answer_tokens = list(map(len, model.to_str_tokens([example['gt']] + wrong_options)))
            all_num_answer_tokens.append(num_answer_tokens)

        full_act_names = [utils.get_act_name(act_name, l) for l in sorted(layers)]

        batch_logits, batch_cache = model.run_with_cache(all_outputs, names_filter=lambda x: x in full_act_names,
                                                         device='cpu',
                                                         padding_side='left')  # logits: (bsz pos vocab) cache: dict
        batch_cache = einops.rearrange([batch_cache[name] for name in full_act_names],
                                       'l b p d -> b l p d').float().cpu()
        batch_cache = einops.rearrange(batch_cache, '(b o) l p d -> b o l p d', o=4)
        batch_cache = batch_cache[:, :, :, [-1], :]

        batch_logits = batch_logits.cpu().float()
        batch_logits = einops.rearrange(batch_logits, '(b o) p v -> b o p v', o=4)

        for i, lg_4 in enumerate(batch_logits):
            num_answer_tokens = all_num_answer_tokens[i]
            num_input_tokens = all_num_input_tokens[i]
            for j, lg in enumerate(lg_4):
                output = all_outputs[i * 4 + j]
                answer_lg = lg[-num_answer_tokens[j] - 1:-1]
                answer_prob = F.softmax(answer_lg, dim=-1)
                answer_target_prob = answer_prob.max(dim=-1).values
                pe = -torch.log(answer_target_prob).sum().item()
                # print(f"pe:{pe}")
                ln_pe = -torch.log(answer_target_prob).mean().item()
                # print(f"ln_pe:{ln_pe}")
                all_pe_u_scores.append(pe)
                all_ln_pe_u_scores.append(ln_pe)

        batch_in_vivo_auroc = []
        for i in range(bsz):
            cache = batch_cache[i]
            u_score = einsum('b l p d, l p d -> b', cache, v_c)
            u_score_z = (u_score - u_score.mean()) / u_score.std()

            u_score = u_score.tolist()
            u_score_z = u_score_z.tolist()

            in_vivo_auroc = roc_auc_score([1, 0, 0, 0], u_score)
            batch_in_vivo_auroc.append(in_vivo_auroc)
            # if u_score[0] > max(u_score[1:]):
            #     batch_in_vivo_auroc.append(1)
            # else:
            #     batch_in_vivo_auroc.append(0)

            c_scores.append(u_score_z[0])
            w_scores.extend(u_score_z[1:])
            labels.extend([1, 0, 0, 0])

            # assert len(u_score) == 4, f"{len(u_score)} {example['options']}"
            u_scores.extend(u_score)
            u_scores_z.extend(u_score_z)

        examples['in_vivo_auroc'] = batch_in_vivo_auroc
        return examples

    new_dst = dst.map(batch_get_result, new_fingerprint=str(time()), batched=True, batch_size=4)

    in_vivo_auroc = sum(new_dst['in_vivo_auroc']) / len(new_dst['in_vivo_auroc'])
    flag = in_vivo_auroc > 0.5
    in_vivo_auroc = in_vivo_auroc if flag else 1 - in_vivo_auroc
    print(f"in-vivo u_score auroc: {in_vivo_auroc}")

    in_vitro_auroc = roc_auc_score(labels, u_scores)
    in_vitro_auroc = in_vitro_auroc if flag else 1 - in_vitro_auroc
    print(f"in-vitro u_score auroc: {in_vitro_auroc}")

    in_vitro_auroc_z = roc_auc_score(labels, u_scores_z)
    in_vitro_auroc_z = in_vitro_auroc_z if flag else 1 - in_vitro_auroc_z
    print(f"in-vitro u_score_z auroc: {in_vitro_auroc_z}")

    in_vitro_pe_auroc = roc_auc_score(labels, all_pe_u_scores)
    print(f"in-vitro pe auroc: {in_vitro_pe_auroc}")

    in_vitro_ln_pe_auroc = roc_auc_score(labels, all_ln_pe_u_scores)
    print(f"in-vitro ln_pe auroc: {in_vitro_ln_pe_auroc}")

    fig.add_trace(go.Histogram(x=c_scores, name='Correct', opacity=0.5, nbinsx=100))
    fig.add_trace(go.Histogram(x=w_scores, name='Wrong', opacity=0.5, nbinsx=100))
    fig.update_layout(barmode='overlay')
    fig.show()
