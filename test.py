# Setup
from utils import *

datasets.disable_progress_bar()
datasets.disable_caching()
torch.set_grad_enabled(False)

# Eval Result Config
model_names_alias = {
    "vicuna-7b-v1.1": "Vicuna-7B",
    "vicuna-13b-v1.1": "Vicuna-13B",
    # "vicuna-33b-v1.3": "Vicuna-33B"
}

dst_names = [
    "sciq",
    "coqa",
    "triviaqa",
    "medmcqa",
    "MedQA-USMLE-4-options",
]

c_metrics = [
    'include',
    'rougel',
    'sentsim',
]

dst_types = [
    "long",
    "short",
]

dst_names_alias = {
    "sciq": "SciQ",
    "coqa": "CoQA",
    "triviaqa": "TriviaQA",
    "medmcqa": "MedMCQA",
    "MedQA-USMLE-4-options": "MedQA"
}

model_dst_acc_map = {
    'vicuna-7b-v1.1':{
        'sciq': 0.661,
        'coqa': 0.626,
        'triviaqa': 0.357,
        'medmcqa': 0.218,
        'MedQA-USMLE-4-options': 0.266
    },
    'vicuna-13b-v1.1':{
        'sciq': 0.715,
        'coqa': 0.648,
        'triviaqa': 0.405,
        'medmcqa': 0.308,
        'MedQA-USMLE-4-options': 0.316
    },
    'vicuna-33b-v1.3':{
        'sciq': 0.831,
        'coqa': 0.688,
        'triviaqa': 0.498,
        'medmcqa': 0.346,
        'MedQA-USMLE-4-options': 0.417
    },
}

# Dst Type: short
# Correctness Metric: include
# Correctness Threshold: 0.3

# vicuna-7b-v1.1 sciq_short Acc: 66.10
# vicuna-7b-v1.1 coqa_short Acc: 62.60
# vicuna-7b-v1.1 triviaqa_short Acc: 35.70
# vicuna-7b-v1.1 medmcqa_short Acc: 21.80
# vicuna-7b-v1.1 MedQA-USMLE-4-options_short Acc: 26.60
# vicuna-13b-v1.1 sciq_short Acc: 71.50
# vicuna-13b-v1.1 coqa_short Acc: 64.80
# vicuna-13b-v1.1 triviaqa_short Acc: 40.50
# vicuna-13b-v1.1 medmcqa_short Acc: 30.80
# vicuna-13b-v1.1 MedQA-USMLE-4-options_short Acc: 31.60
# vicuna-33b-v1.3 sciq_short Acc: 83.10
# vicuna-33b-v1.3 coqa_short Acc: 68.80
# vicuna-33b-v1.3 triviaqa_short Acc: 49.80
# vicuna-33b-v1.3 medmcqa_short Acc: 34.60
# vicuna-33b-v1.3 MedQA-USMLE-4-options_short Acc: 41.70

# Dst Type: long
# Correctness Metric: include
# Correctness Threshold: 0.3

# vicuna-7b-v1.1 sciq_long Acc: 64.20
# vicuna-7b-v1.1 coqa_long Acc: 63.20
# vicuna-7b-v1.1 triviaqa_long Acc: 36.20
# vicuna-7b-v1.1 medmcqa_long Acc: 17.00
# vicuna-7b-v1.1 MedQA-USMLE-4-options_long Acc: 19.50
# vicuna-13b-v1.1 sciq_long Acc: 74.80
# vicuna-13b-v1.1 coqa_long Acc: 61.60
# vicuna-13b-v1.1 triviaqa_long Acc: 37.60
# vicuna-13b-v1.1 medmcqa_long Acc: 23.50
# vicuna-13b-v1.1 MedQA-USMLE-4-options_long Acc: 29.30
# vicuna-33b-v1.3 sciq_long Acc: 56.10
# vicuna-33b-v1.3 coqa_long Acc: 64.60
# vicuna-33b-v1.3 triviaqa_long Acc: 44.60
# vicuna-33b-v1.3 medmcqa_long Acc: 16.90
# vicuna-33b-v1.3 MedQA-USMLE-4-options_long Acc: 31.10

u_metric_alias = {
    "u_score_pe": "PE",
    "u_score_ln_pe": "LN-PE",
    "u_score_token_sar": "TokenSAR",
    "u_score_sent_sar": "SentSAR",
    # "u_score_sar": "SAR",
    "u_score_ls": "LS",
    "u_score_se": "SE",
    "u_score_ours_mlp_small_attn_token_all": "AUP(Ours)",
    "u_score_ours_mlp_small_last_inp": "KSP",
}

u_metric_class = {
    "Single Inference Methods":[
        "PE",
        "LN-PE",
        "TokenSAR",
    ],
    "Multi Inference Methods":[
        "SentSAR",
        # "SAR",
        "LS",
        "SE"
    ],
    "Internal State Methods":[
        "KSP",
        "AUP(Ours)"
    ]
}

def get_cached_result_path(model_name, dst_name, dst_type, dst_split):
    return f"cached_results/{model_name}/{dst_type}/{dst_name}_{dst_split}"


def get_eval_baseline_result_path(model_name, dst_name, dst_type):
    return f"baseline_eval_results/{model_name}/{dst_name}_{dst_type}"


def get_eval_ours_result_path(model_name, dst_name, dst_type, c_metric, vc_type):
    return f"ours_eval_results/{model_name}/{c_metric}//v_c_{dst_name}_{dst_type}_{vc_type}.pth/{dst_name}_{dst_type}"


def get_eval_cross_result_path(model_name, train_dst_name, train_dst_type, test_dst_name, test_dst_type, c_metric, vc_type):
    return f"cross_eval_results/{model_name}/{c_metric}/v_c_{train_dst_name}_{train_dst_type}_{vc_type}.pth/{test_dst_name}_{test_dst_type}"


def get_c_th_by_acc(test_dst, c_metric, acc):
    sorted_c_scores = sorted(list(test_dst[c_metric]), reverse=True)
    c_th = sorted_c_scores[int(len(sorted_c_scores) * acc)]
    if c_th == 0:
        c_th += 0.01
    elif c_th == 1:
        c_th -= 0.01
    return c_th

def get_acc_by_c_th(test_dst, c_metric, c_th):
    return sum([1 if s > c_th else 0 for s in test_dst[c_metric]]) / len(test_dst)

# Load LLM Model
model_name = 'vicuna-7b-v1.1'
hooked_transformer_name = get_hooked_transformer_name(model_name)
hf_model_path = os.path.join(os.environ["my_models_dir"], model_name)
hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
hf_tokenizer.pad_token_id = hf_tokenizer.eos_token_id
with LoadWoInit():
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)
model = HookedTransformer.from_pretrained_no_processing(hooked_transformer_name, dtype='bfloat16', hf_model=hf_model, tokenizer=hf_tokenizer, default_padding_side='left')

# Show Sampling Generation Sample
test_dst = Dataset.load_from_disk(get_cached_result_path('vicuna-7b-v1.1', 'sciq', 'long', 'validation'))
test_dst = test_dst.filter(lambda x: x['gt'] == 'mechanical')
idx = 0
sample_num = 10
temperature = 1.0
from time import time

# Generate
example = test_dst[idx]
for k in ['input', 'gt',]:
    print(f"{k}: {example[k]}")

prompt = example['input']
all_ans = []
start = time()
for i in tqdm(range(sample_num)):
    inp = model.to_tokens(prompt)
    out = model.generate(inp, max_new_tokens=128, do_sample=True, temperature=temperature, return_type='tensor', verbose=False)
    prob = F.softmax(model(out), dim=-1)
    ans_prob = prob[0, inp.shape[-1]-1:-1][range(len(out[0, inp.shape[-1]:])), out[0, inp.shape[-1]:]].tolist()
    ans_prob = list(map(lambda x: str(x)[:5], ans_prob))
    ans = out[0, inp.shape[-1]:]
    answer = model.to_string(ans)
    print(f"Generated Answer {i}: {answer}\n{ans_prob}")
    all_ans.append(answer)
print(time()-start)