# %%
# Unofficial Inplementation of In-Context-Vector with transformer_lens Library
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix
import torch
import torch.nn as nn
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import List, Tuple
import os
import socket

# %%
hostname = socket.gethostname()
print("当前主机名是：" + hostname)
IS_CLUSTER = "SH-IDC" in hostname

if IS_CLUSTER:
    model_path = os.path.join(os.environ['my_models_dir'], 'llama-7b')
    llama_7b_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = HookedTransformer.from_pretrained_no_processing("llama-7b-hf", hf_model=llama_7b_model, tokenizer=tokenizer, dtype='float16', default_padding_side='left')
else:
    model = HookedTransformer.from_pretrained_no_processing("gpt2-xl", dtype='float16', default_padding_side='left')
dataset = load_dataset("s-nlp/paradetox")

# %%
ACT_NAME = 'resid_post'
PROMPT="Instruction: Please paraphrase the following sentence.\n"
FS_TEMPLATE = "Sentence:{}\nParaphrase:{}"

# Here we copy the PCA code from the original repo
def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v

class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_

def get_icv(model:HookedTransformer, positive_sentences, negative_sentences):
    pos_tokens = model.to_tokens(positive_sentences, padding_side='left')
    neg_tokens = model.to_tokens(negative_sentences, padding_side='left')
    names_filter = lambda x : x.startswith("blocks.") and x.endswith(ACT_NAME)
    pos_logits, pos_cache = model.run_with_cache(pos_tokens, names_filter=names_filter)
    neg_logits, neg_cache = model.run_with_cache(neg_tokens, names_filter=names_filter)
    pos_vectors = einops.rearrange([pos_cache[utils.get_act_name(ACT_NAME, l)][:,-1,:] for l in range(model.cfg.n_layers)], 'l b d -> l b d')
    neg_vectors = einops.rearrange([neg_cache[utils.get_act_name(ACT_NAME, l)][:,-1,:] for l in range(model.cfg.n_layers)], 'l b d -> l b d')
    fit_data = einops.rearrange(pos_vectors - neg_vectors, 'l b d -> b (l d)')
    pca = PCA(n_components=1).to(fit_data.device).fit(fit_data.float())
    direction = (pca.components_.sum(dim=0,keepdim=True) + pca.mean_).mean(0)
    icv = direction.view(model.cfg.n_layers, -1)
    return icv

def apply_icv(model:HookedTransformer, icv, lamb=0.1):
    model.reset_hooks()
    def residual_stream_edit_hook(
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        hook: HookPoint,
        layer: int
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        original_norm = torch.norm(resid_pre, dim=-1, keepdim=True)
        resid_pre += einops.repeat(icv[layer], 'd_model -> batch pos d_model', batch=resid_pre.shape[0], pos=resid_pre.shape[1]) * lamb
        new_norm = torch.norm(resid_pre, dim=-1, keepdim=True)
        resid_pre = resid_pre / new_norm * original_norm
        return resid_pre
    for l in range(model.cfg.n_layers):
        model.blocks[l].hook_resid_pre.add_hook(partial(residual_stream_edit_hook, layer=l))
    return model

def compare_all(model:HookedTransformer, test_sentences, positive_sentences, negative_sentences):
    model.reset_hooks()
    zs_input = [PROMPT+FS_TEMPLATE.format(s,'') for s in test_sentences]
    fs_examples = '\n'.join([FS_TEMPLATE.format(s1,s2) for s1,s2 in zip(negative_sentences, positive_sentences)])
    fs_input = [PROMPT+fs_examples+'\n'+FS_TEMPLATE.format(s,'') for s in test_sentences]
    
    zs_tokens = model.to_tokens(zs_input, padding_side='left')
    fs_tokens = model.to_tokens(fs_input, padding_side='left')
    gen_configs = dict(max_new_tokens=30, temperature=0, top_k=1, top_p=1, do_sample=False)
    
    zs_output = model.generate(zs_tokens, **gen_configs)
    zs_output = model.to_string(zs_output)
    fs_output = model.generate(fs_tokens, **gen_configs)
    fs_output = model.to_string(fs_output)
        
    icv = get_icv(model, positive_sentences, negative_sentences)
    
    model = apply_icv(model, icv)
    icv_output = model.generate(zs_tokens, **gen_configs)
    model.reset_hooks()    
    
    icv_output = model.to_string(icv_output)
    pad_token = model.tokenizer.pad_token
    bos_token = model.tokenizer.bos_token
    wash = lambda text, pattern : text.replace(pattern,'').replace(pad_token,'').replace(bos_token,'').strip()
    
    for zi, fi, zo, fo, io in zip(zs_input,fs_input, zs_output,fs_output,icv_output):
        print(f"ZS PROMPT:\n{zi}\n")
        print(f"FS PROMPT:\n{fi}\n")
        print(f"ZS:\n{wash(zo, zi)}\n")
        print(f"FS:\n{wash(fo, fi)}\n")
        print(f"ICV:\n{wash(io, zi)}\n")
        print("\n\n")

# %%
num_shots = 10
num_test = 20

positive_sentences = dataset['train']['en_neutral_comment'][:num_shots]
negative_sentences = dataset['train']['en_toxic_comment'][:num_shots]
test_sentences = dataset['train']['en_toxic_comment'][-num_test:]
# print(positive_sentences)
# print(negative_sentences)
# print(test_sentences)

# %%
compare_all(model, test_sentences, positive_sentences, negative_sentences)


