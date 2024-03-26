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
import inspect
from utils import *

datasets.disable_caching()
torch.set_grad_enabled(False)
print_sys_info()

# All Config Variables
se_bert_name = "microsoft/deberta-large-mnli"
sentsim_bert_name = "all-MiniLM-L6-v2"
sar_bert_name = 'all-MiniLM-L6-v2'
all_score_func = ["mean", "last"]
all_label_type = ["soft", "hard"]
all_c_metric = ["rougel", "sentsim"]
all_u_func = ["len", "pe", "sar", "ls", "se", "ours"]

def evaluate(model_name,
             test_dst_path,
             c_metric="all",
             u_func="all",
             ):
    # print args
    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    for arg in args:
        print(f"{arg} = {values[arg]}")

    # Load LLM Model
    hooked_transformer_name = get_hooked_transformer_name(model_name)
    hf_model_path = os.path.join(os.environ["my_models_dir"], model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    with LoadWoInit():
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)
    model = HookedTransformer.from_pretrained_no_processing(hooked_transformer_name, dtype='bfloat16', hf_model=hf_model, tokenizer=hf_tokenizer, default_padding_side='left')

    # Load Aux Models For SE and SAR
    nli_pipe = pipeline("text-classification", model=se_bert_name, device=0)
    sar_bert = SentenceTransformer(sar_bert_name)

    dst_name = test_dst_path.split("/")[-1].split("_")[1]
    print(f"dst_name:{dst_name}")

    if "short" in test_dst_path:
        dst_type = "short"
    elif "long" in test_dst_path:
        dst_type = "long"
    else:
        raise ValueError("dst_type not supported")

    print(f"dst_type:{dst_type}")

    def _get_answer_prob(inp, out, prob):
        num_input_tokens = len(model.to_str_tokens(inp))
        output_tokens = model.to_tokens(out, move_to_device=False)[0].tolist()
        if len(output_tokens) == num_input_tokens:
            return []
        answer_tokens = output_tokens[num_input_tokens:]
        answer_prob = prob[num_input_tokens - 1:-1, :]
        answer_prob = answer_prob[range(len(answer_tokens)), answer_tokens]
        answer_prob = answer_prob.tolist()
        return answer_prob

    def get_answer_prob(examples):
        batch_answer_prob = []
        batch_prob = F.softmax(model(examples['washed_output'], padding_side='right'), dim=-1)  # logits: (bsz pos vocab)

        for i in range(len(examples['washed_output'])):
            answer_prob = _get_answer_prob(examples['input'][i], examples['washed_output'][i], batch_prob[i])
            batch_answer_prob.append(answer_prob)

        examples['answer_prob'] = batch_answer_prob
        return examples

    def get_uncertainty_score_token_pe_all(example):
        if not example['answer_prob']:
            example['u_score_pe_all'] = []
            example['u_score_pe'] = 0
            example['u_score_ln_pe'] = 0
        answer_prob = torch.tensor(example['answer_prob'], dtype=torch.float)
        neg_logp = -torch.log(answer_prob)
        example['u_score_pe_all'] = neg_logp.tolist()
        example['u_score_pe'] = neg_logp.sum().item()
        example['u_score_ln_pe'] = neg_logp.mean().item()

        return example

    def get_uncertainty_score_ls(example):
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
        return example

    def get_uncertainty_score_se(example, nli_pipe):
        # Sample Answers
        sampled_outputs = example['washed_sampled_answer']

        # Bidirectional Entailment Clustering
        meanings = [[sampled_outputs[0]]]
        seqs = sampled_outputs[1:]

        for s in seqs:
            for c in meanings:
                s_c = c[0]
                tmp = "[CLS] {s1} [SEP] {s2} [CLS]"
                res = nli_pipe([tmp.format(s1=s, s2=s_c), tmp.format(s1=s_c, s2=s)])
                if res[0]['label'] == 'ENTAILMENT' and res[1]['label'] == 'ENTAILMENT':
                    c.append(s)
                    break
                else:
                    meanings.append([s])

        # Calculate Semantic Entropy
        pcs = []
        for c in meanings:
            pc = torch.tensor([0.], dtype=torch.float)
            for s in c:
                prob = F.softmax(model(s), dim=-1)[0]
                answer_prob = _get_answer_prob(example['input'], s, prob)
                ps = torch.prod(torch.tensor(answer_prob, dtype=torch.float))
                pc += ps
            pcs.append(pc)
        pcs = torch.tensor(pcs)

        example['u_score_se'] = -(torch.log(pcs) * pcs).sum().item()
        return example

    def get_uncertainty_score_token_sar(example, sar_bert):
        if example['washed_answer'] == "":
            example['u_score_sar'] = 0
            return example
        num_input_tokens = len(model.to_str_tokens(example['input']))
        num_output_tokens = len(model.to_str_tokens(example['washed_output']))
        orig_embedding = sar_bert.encode(example['washed_output'], convert_to_tensor=True)
        neg_logp = -torch.log(torch.tensor(example['answer_prob']))

        input_tokens = model.to_tokens(example['washed_output'], move_to_device=False)[0].tolist()
        start, end = num_input_tokens, num_output_tokens
        new_input_strings = []
        for j in range(start, end):
            new_input_tokens = input_tokens[:j] + input_tokens[j + 1:]
            new_input_string = model.to_string(new_input_tokens)
            new_input_strings.append(new_input_string)
        new_embeddings = sar_bert.encode(new_input_strings, convert_to_tensor=True)
        sim = st_util.cos_sim(orig_embedding, new_embeddings)[0].cpu()

        weights = 1 - sim
        weights = F.softmax(weights, dim=0) * len(weights)
        sar_score = einsum('s, s ->', neg_logp, weights).item()

        example['u_score_sar'] = sar_score
        return example

    def get_uncertainty_score_len(example):
        example['u_score_len'] = example['num_answer_tokens']
        return example

    def get_uncertainty_score_ours_all(examples, v_c, score_func, label_type):
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
            for th in th_range:
                aurocs.append(get_auroc(test_dst, u_metric, c_metric, th))
            fig.add_trace(go.Scatter(x=th_range, y=aurocs, mode='lines+markers+text', name=f"{u_metric}", text=[f"{a:.4f}" for a in aurocs], textposition="top center"))
        fig.update_layout(title=f"AUROC/{c_metric}-Threshold Curve", xaxis_title=f"{c_metric}-Threshold", yaxis_title="AUROC", width=2000, height=1000)

        return fig

    def get_vc_path(dst_name, dst_type, model_name, label_type, score_func):
        vc_path = "/mnt/petrelfs/guoyiqiu/coding/trainable_uncertainty/models"
        fs = os.listdir(f"{vc_path}")
        new_fs = []
        for f in fs:
            if dst_name in f and model_name in f and dst_type in f and label_type in f and score_func in f:
                new_fs.append(f)
        assert len(new_fs) == 1
        return f"{vc_path}/{new_fs[0]}"

    test_dst = Dataset.load_from_disk(test_dst_path)

    # test_dst = test_dst.select(range(10))

    if dst_type == "long":
        first_sentence_only = True
    else:
        first_sentence_only = False

    print("Running wash_answer")
    test_dst = test_dst.map(partial(wash_answer, tokenizer=hf_tokenizer, first_sentence_only=first_sentence_only), new_fingerprint=str(time()))

    print("Running get_num_tokens")
    test_dst = test_dst.map(partial(get_num_tokens, tokenizer=hf_tokenizer), batched=True, batch_size=8, new_fingerprint=str(time()))

    if c_metric == "all":
        c_metrics = all_c_metric
    elif c_metric in all_c_metric:
        c_metrics = [c_metric]
    else:
        raise ValueError(f"c_metric {c_metric} not supported")

    if "rougel" in c_metrics:
        print("Running get_rougel")
        test_dst = test_dst.map(get_rougel, new_fingerprint=str(time()))

    if "sentsim" in c_metrics:
        print("Running get_sentsim")
        st_model = SentenceTransformer(sentsim_bert_name)
        test_dst = test_dst.map(partial(get_sentsim, st_model=st_model), new_fingerprint=str(time()))

    print("Running get_answer_prob")
    test_dst = test_dst.map(get_answer_prob, batched=True, batch_size=8, new_fingerprint=str(time()))

    keys = (['options'] if test_dst[0].get('options') else []) + ['question', 'washed_answer', 'gt', 'num_answer_tokens'] + c_metrics
    for i in range(10):
        for k in keys:
            print(f"{k}:{test_dst[-i][k]}")
        print()

    if u_func == "all":
        u_funcs = all_u_func
    elif u_func in all_u_func:
        u_funcs = [u_func]
    else:
        raise ValueError(f"u_func {u_func} not supported")

    # Run all baseline func

    if "len" in u_funcs:
        print("Running get_uncertainty_score_len")
        test_dst = test_dst.map(get_uncertainty_score_len, new_fingerprint=str(time()))

    if "pe" in u_funcs:
        print("Running get_uncertainty_score_pe_all")
        pe_all_func = partial(get_uncertainty_score_token_pe_all)
        test_dst = test_dst.map(pe_all_func, new_fingerprint=str(time()))

    if "sar" in u_funcs:
        print("Running get_uncertainty_score_sar")
        sar_func = partial(get_uncertainty_score_token_sar, sar_bert=sar_bert)
        test_dst = test_dst.map(sar_func, new_fingerprint=str(time()))

    if "ls" in u_funcs:
        print("Running get_uncertainty_score_ls")
        ls_func = partial(get_uncertainty_score_ls)
        test_dst = test_dst.map(ls_func, new_fingerprint=str(time()))

    if "se" in u_funcs:
        print("Running get_uncertainty_score_se")
        se_func = partial(get_uncertainty_score_se, nli_pipe=nli_pipe)
        test_dst = test_dst.map(se_func, new_fingerprint=str(time()))

    if "ours" in u_funcs:
        # Run our func
        CACHED_LAYERS = list(range(0, model.cfg.n_layers))
        CACHED_ACT_NAME = 'resid_post'
        full_act_names = [utils.get_act_name(CACHED_ACT_NAME, l) for l in sorted(CACHED_LAYERS)]
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
        for score_func in all_score_func:
            for label_type in all_label_type:
                vc_path = get_vc_path(dst_name, dst_type, model_name, label_type, score_func)
                v_c.load_state_dict(torch.load(vc_path))
                for v in v_c.values():
                    v.eval()
                    v.to(model.cfg.dtype).to(model.cfg.device)
                    for p in v.parameters():
                        p.requires_grad = False

                print(f"Running get_uncertainty_score_ours_{score_func}_{label_type}")
                ours_func = partial(get_uncertainty_score_ours_all, v_c=v_c, score_func=score_func, label_type=label_type)
                test_dst = test_dst.map(ours_func, batched=True, batch_size=8, new_fingerprint=str(time()))

    print(f"average num answer tokens:{np.mean(test_dst['u_score_len'])}")
    print(f"average sample answer rougel:{np.mean(test_dst['u_score_ls'])}")

    # Save the result
    base_dir = "/mnt/petrelfs/guoyiqiu/coding/gpt_mi/eval_results"
    save_base_name = f"{base_dir}/{dst_name}_{dst_type}_{model_name}"
    os.makedirs(save_base_name, exist_ok=True)

    u_metrics = [k for k in test_dst[0].keys() if k.startswith("u_score") and not k.endswith("all")]
    for c_metric in all_c_metric:
        fig = plot_th_curve(test_dst, u_metrics, c_metric)
        fig.write_html(f"{save_base_name}/{c_metric}_th_curve.html")
        fig.write_image(f"{save_base_name}/{c_metric}_th_curve.png")

    test_dst.save_to_disk(save_base_name)
    print(f"save all results to {save_base_name}")


if __name__ == "__main__":
    fire.Fire(evaluate)
