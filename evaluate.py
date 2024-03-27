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
all_u_metric = ["len", "pe", "sar", "ls", "se", "ours"]
eval_batch_size = 8
t_sar=0.001

def evaluate(model_name,
             test_dst_path,
             c_metric="all",
             u_metric="all",
             ):
    # print args
    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    for arg in args:
        print(f"{arg} = {values[arg]}")
    dst_name = test_dst_path.split("/")[-1].split("_")[1]
    print(f"dst_name:{dst_name}")

    if "short" in test_dst_path:
        dst_type = "short"
    elif "long" in test_dst_path:
        dst_type = "long"
    else:
        raise ValueError("dst_type not supported")
    print(f"dst_type:{dst_type}")

    # Load LLM Model
    hooked_transformer_name = get_hooked_transformer_name(model_name)
    hf_model_path = os.path.join(os.environ["my_models_dir"], model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    with LoadWoInit():
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)
    model = HookedTransformer.from_pretrained_no_processing(hooked_transformer_name, dtype='bfloat16', hf_model=hf_model, tokenizer=hf_tokenizer, default_padding_side='left')

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
        bsz = len(examples['input'])
        with Timer() as timer:
            batch_logits = model(examples['washed_output'], padding_side='right')

        batch_prob = F.softmax(batch_logits, dim=-1)  # prob: (bsz pos vocab)

        for i in range(len(examples['washed_output'])):
            answer_prob = _get_answer_prob(examples['input'][i], examples['washed_output'][i], batch_prob[i])
            batch_answer_prob.append(answer_prob)

        examples['answer_prob'] = batch_answer_prob
        examples['time_fwd'] = [timer.t / bsz for i in range(bsz)]
        return examples

    def get_sampled_answer_prob(example):
        batch_answer_prob = []
        washed_sampled_output = example['washed_sampled_output']
        washed_sampled_output_unique = list(set(washed_sampled_output))
        batch_prob = F.softmax(model(washed_sampled_output_unique, padding_side='right'), dim=-1)  # logits: (bsz pos vocab)

        for i in range(len(example['washed_sampled_output'])):
            inp = example['input']
            out = example['washed_sampled_output'][i]
            prob = batch_prob[washed_sampled_output_unique.index(out)]
            answer_prob = _get_answer_prob(inp, out, prob)
            batch_answer_prob.append(answer_prob)

        example['sampled_answer_prob'] = batch_answer_prob
        return example

    def get_uncertainty_score_token_pe_all(examples):
        with Timer() as timer:
            if not examples.get('answer_prob'):
                examples = get_answer_prob(examples)
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

    def get_uncertainty_score_se(example, nli_pipe):
        # Sample Answers
        washed_sampled_answer = example['washed_sampled_answer']
        if not example.get('sampled_answer_prob'):
            example = get_sampled_answer_prob(example)
        with Timer() as timer:
            # Bidirectional Entailment Clustering
            meanings = [[washed_sampled_answer[0]]]
            seqs = washed_sampled_answer[1:]

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
                    idx = example['washed_sampled_answer'].index(s)
                    answer_prob = example['sampled_answer_prob'][idx]
                    ps = torch.prod(torch.tensor(answer_prob, dtype=torch.float))
                    pc += ps
                pcs.append(pc)
            pcs = torch.tensor(pcs)

            example['u_score_se'] = -(torch.log(pcs) * pcs).sum().item()
        example['time_se'] = timer.t
        return example

    def get_uncertainty_score_sar_all(example, sar_bert, T):
        if not example.get('answer_prob'):
            example = get_answer_prob(example)

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
            example = get_sampled_answer_prob(example)

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

    def get_uncertainty_score_ours_all(examples, v_c, score_func, label_type):
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

        examples[f'time_ours_{score_func}_{label_type}'] = [timer.t / bsz - examples[i]['time_forward'] for i in range(bsz)]
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
    test_dst = test_dst.map(partial(get_num_tokens, tokenizer=hf_tokenizer), batched=True, batch_size=eval_batch_size, new_fingerprint=str(time()))

    if isinstance(c_metric, str):
        if c_metric == "all":
            c_metrics = all_c_metric
        elif "," in c_metric:
            c_metrics = [c.strip() for c in c_metric.split(",")]
        elif c_metric in all_c_metric:
            c_metrics = [c_metric]
        else:
            raise ValueError(f"c_metric {c_metric} not supported")
    elif isinstance(c_metric, tuple):
        c_metrics = list(c_metric)
    else:
        raise ValueError(f"c_metric {c_metric} not supported")

    for c in c_metrics:
        if c not in all_c_metric:
            raise ValueError(f"c_metric {c} not supported")

    print('c_metrics: ', c_metrics)

    if "rougel" in c_metrics:
        print("Running get_rougel")
        test_dst = test_dst.map(get_rougel, new_fingerprint=str(time()))

    if "sentsim" in c_metrics:
        print("Running get_sentsim")
        st_model = SentenceTransformer(sentsim_bert_name)
        test_dst = test_dst.map(partial(get_sentsim, st_model=st_model), batched=True, batch_size=eval_batch_size, new_fingerprint=str(time()))

    keys = (['options'] if test_dst[0].get('options') else []) + ['question', 'washed_answer', 'gt', 'num_answer_tokens'] + c_metrics
    for i in range(10):
        for k in keys:
            print(f"{k}:{test_dst[-i][k]}")
        print()

    if isinstance(u_metric, str):
        if u_metric == "all":
            u_metrics = all_u_metric
        elif "," in u_metric:
            u_metrics = [c.strip() for c in u_metric.split(",")]
        elif u_metric in all_u_metric:
            u_metrics = [u_metric]
        else:
            raise ValueError(f"u_metric {u_metric} not supported")
    elif isinstance(u_metric, tuple):
        u_metrics = list(u_metric)
    else:
        raise ValueError(f"u_metric {u_metric} not supported")

    for c in u_metrics:
        if c not in all_u_metric:
            raise ValueError(f"u_metric {c} not supported")

    print('u_metrics: ', u_metrics)
    if set(u_metrics) & {'pe', 'sar'}:
        print("Running get_answer_prob")
        test_dst = test_dst.map(get_answer_prob, batched=True, batch_size=eval_batch_size, new_fingerprint=str(time()))

    if set(u_metrics) & {'se', 'sar'}:
        print("Running get_sampled_answer_prob")
        test_dst = test_dst.map(get_sampled_answer_prob, new_fingerprint=str(time()))

    if "len" in u_metrics:
        print("Running get_uncertainty_score_len")
        test_dst = test_dst.map(get_uncertainty_score_len, new_fingerprint=str(time()))
        print(f"average num answer tokens:{np.mean(test_dst['u_score_len'])}")

    if "pe" in u_metrics:
        print("Running get_uncertainty_score_pe_all")
        pe_all_func = partial(get_uncertainty_score_token_pe_all)
        test_dst = test_dst.map(pe_all_func, batched=True, batch_size=eval_batch_size, new_fingerprint=str(time()))
        print(f"time_pe:{sum(test_dst['time_pe'])}")

    if "sar" in u_metrics:
        print("Running get_uncertainty_score_sar")
        sar_bert = SentenceTransformer(sar_bert_name)
        sar_func = partial(get_uncertainty_score_sar_all, sar_bert=sar_bert, T=t_sar)
        test_dst = test_dst.map(sar_func, new_fingerprint=str(time()))
        print(f'time_token_sar:{sum(test_dst["time_token_sar"])}')
        print(f'time_sent_sar:{sum(test_dst["time_sent_sar"])}')
        print(f'time_sar:{sum(test_dst["time_sar"])}')

    if "ls" in u_metrics:
        print("Running get_uncertainty_score_ls")
        ls_func = partial(get_uncertainty_score_ls)
        test_dst = test_dst.map(ls_func, new_fingerprint=str(time()))
        print(f"average sample answer rougel:{np.mean(test_dst['u_score_ls'])}")
        print(f"time_ls:{sum(test_dst['time_ls'])}")

    if "se" in u_metrics:
        print("Running get_uncertainty_score_se")
        nli_pipe = pipeline("text-classification", model=se_bert_name, device=0)
        se_func = partial(get_uncertainty_score_se, nli_pipe=nli_pipe)
        test_dst = test_dst.map(se_func, new_fingerprint=str(time()))
        print(f"time_ls:{sum(test_dst['time_ls'])}")

    if "ours" in u_metrics:
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
                test_dst = test_dst.map(ours_func, batched=True, batch_size=eval_batch_size, new_fingerprint=str(time()))
                print(f"time_ours_{score_func}_{label_type}:{sum(test_dst[f'time_ours_{score_func}_{label_type}'])}")

    keys = (['options'] if test_dst[0].get('options') else []) + ['question', 'washed_answer', 'gt', 'num_answer_tokens'] + c_metrics + [k for k in test_dst[0].keys() if
                                                                                                                                         k.startswith("u_score") and not k.endswith("all")]
    for i in range(10):
        for k in keys:
            print(f"{k}:{test_dst[-i][k]}")
        print()

    # Save the result
    base_dir = "/mnt/petrelfs/guoyiqiu/coding/trainable_uncertainty/eval_results"
    save_base_name = f"{base_dir}/{dst_name}_{dst_type}_{model_name}"
    os.makedirs(save_base_name, exist_ok=True)

    u_metrics = [k for k in test_dst[0].keys() if k.startswith("u_score") and not k.endswith("all")]
    for c_metric in c_metrics:
        fig = plot_th_curve(test_dst, u_metrics, c_metric)
        fig.write_html(f"{save_base_name}_{c_metric}_curve.html")
        fig.write_image(f"{save_base_name}_{c_metric}_curve.png")

    # test_dst.save_to_disk(save_base_name)
    print(f"save all results to {save_base_name}")


if __name__ == "__main__":
    fire.Fire(evaluate)
