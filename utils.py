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