import inspect
from utils import *

datasets.disable_caching()
torch.set_grad_enabled(False)
print_sys_info()
# All Config Variables
se_bert_name = "microsoft/deberta-large-mnli"
sentsim_bert_name = "all-MiniLM-L6-v2"
sar_bert_name = 'all-MiniLM-L6-v2'
all_c_metric = ["rougel", "sentsim", "include"]
all_u_metric = ["len", "pe", "sar", "ls", "se", "ours"]
eval_batch_size = 8
t_sar = 0.001

def parse_metric(c_metric, all_c_metric):
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

    return c_metrics


def get_vc_path(dst_name, dst_type, model_name, label_name, label_type, score_func):
    return f"models/{model_name}/{label_name}/v_c_{dst_name}_{dst_type}_{score_func}_{label_type}_best.pth"


def evaluate(
        model_name,
        dst_name,
        dst_type,
        c_metric="all",
        u_metric="all",
        custom_vc_path=None,
        custom_save_path=None,
        max_val_data_size=1000,
        merge_existing_result=False
):
    # print args
    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    for arg in args:
        print(f"{arg} = {values[arg]}")

    c_metrics = parse_metric(c_metric, all_c_metric)
    print('c_metrics: ', c_metrics)
    u_metrics = parse_metric(u_metric, all_u_metric)
    print('u_metrics: ', u_metrics)

    # Load Test Dst
    test_dst_path = f'cached_results/{model_name}/{dst_type}/{dst_name}_validation'
    if not os.path.exists(test_dst_path):
        raise ValueError(f"cached results not found at {test_dst_path}")
    test_dst = Dataset.load_from_disk(test_dst_path)

    # Save Config
    default_save_path = f"eval_results/{model_name}/{dst_name}_{dst_type}"
    save_path = custom_save_path if custom_save_path else default_save_path
    os.makedirs(save_path, exist_ok=True)

    if merge_existing_result and os.path.exists(f"{save_path}/dataset_info.json"):
        print(f"Merging existing result at {save_path}")
        existing_dst = Dataset.load_from_disk(save_path)
        for k in existing_dst.column_names:
            if k not in test_dst.column_names:
                test_dst = test_dst.add_column(name=k, column=existing_dst[k])
        print(f"Existing result merged, added keys:{test_dst.column_names}")

    # Load LLM Model
    hooked_transformer_name = get_hooked_transformer_name(model_name)
    hf_model_path = os.path.join(os.environ["my_models_dir"], model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    with LoadWoInit():
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)
    model = HookedTransformer.from_pretrained_no_processing(hooked_transformer_name, dtype='bfloat16', hf_model=hf_model, tokenizer=hf_tokenizer, default_padding_side='left')

    if max_val_data_size is not None:
        test_dst = test_dst.select(range(min(len(test_dst), max_val_data_size)))

    print("Running wash_answer")
    test_dst = test_dst.map(partial(wash_answer, tokenizer=hf_tokenizer, first_sentence_only=(dst_type == "long")), new_fingerprint=str(time()))

    print("Running get_num_tokens")
    test_dst = test_dst.map(partial(get_num_tokens, tokenizer=hf_tokenizer), batched=True, batch_size=eval_batch_size, new_fingerprint=str(time()))

    if "rougel" in c_metrics:
        print("Running get_rougel")
        test_dst = test_dst.map(get_rougel, new_fingerprint=str(time()))

    if "sentsim" in c_metrics:
        print("Running get_sentsim")
        st_model = SentenceTransformer(sentsim_bert_name)
        test_dst = test_dst.map(partial(get_sentsim, st_model=st_model), batched=True, batch_size=eval_batch_size, new_fingerprint=str(time()))

    if "include" in c_metrics:
        print("Running get_include")
        test_dst = test_dst.map(get_include, new_fingerprint=str(time()))

    keys = (['options'] if test_dst[0].get('options') else []) + ['question', 'washed_answer', 'gt', 'num_answer_tokens'] + c_metrics
    for i in range(10):
        for k in keys:
            print(f"{k}:{test_dst[-i][k]}")
        print()

    if set(u_metrics) & {'pe', 'sar'}:
        if "answer_prob" not in test_dst.column_names:
            print("Running get_answer_prob")
            test_dst = test_dst.map(get_answer_prob, fn_kwargs=dict(model=model), batched=True, batch_size=eval_batch_size, new_fingerprint=str(time()))

    if set(u_metrics) & {'se', 'sar'}:
        if "sampled_answer_prob" not in test_dst.column_names:
            print("Running get_sampled_answer_prob")
            test_dst = test_dst.map(get_sampled_answer_prob, fn_kwargs=dict(model=model), new_fingerprint=str(time()))

    if "len" in u_metrics:
        print("Running get_uncertainty_score_len")
        test_dst = test_dst.map(get_uncertainty_score_len, new_fingerprint=str(time()))
        print(f"average num answer tokens:{np.mean(test_dst['u_score_len'])}")

    if "pe" in u_metrics:
        print("Running get_uncertainty_score_pe_all")
        test_dst = test_dst.map(get_uncertainty_score_token_pe_all, fn_kwargs=dict(model=model), batched=True, batch_size=eval_batch_size, new_fingerprint=str(time()))
        print(f"time_pe:{sum(test_dst['time_pe'])}")

    if "sar" in u_metrics:
        print("Running get_uncertainty_score_sar")
        sar_bert = SentenceTransformer(sar_bert_name)
        sar_func = partial(get_uncertainty_score_sar_all, sar_bert=sar_bert, T=t_sar, model=model)
        test_dst = test_dst.map(sar_func, new_fingerprint=str(time()))
        print(f'time_token_sar:{sum(test_dst["time_token_sar"])}')
        print(f'time_sent_sar:{sum(test_dst["time_sent_sar"])}')
        print(f'time_sar:{sum(test_dst["time_sar"])}')

    if "ls" in u_metrics:
        print("Running get_uncertainty_score_ls")
        test_dst = test_dst.map(get_uncertainty_score_ls, new_fingerprint=str(time()))
        print(f"average sample answer rougel:{np.mean(test_dst['u_score_ls'])}")
        print(f"time_ls:{sum(test_dst['time_ls'])}")

    if "se" in u_metrics:
        print("Running get_uncertainty_score_se")
        nli_pipe = pipeline("text-classification", model=se_bert_name, device=0)
        test_dst = test_dst.map(get_uncertainty_score_se, fn_kwargs=dict(nli_pipe=nli_pipe, model=model), new_fingerprint=str(time()))
        print(f"time_se:{sum(test_dst['time_se'])}")

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

        all_label_name = ['rougel', 'sentsim', 'include']
        all_score_func = ["mean", "last"]
        all_label_type = ["soft"]

        if custom_vc_path:
            print(f"find custom_vc_path: {custom_vc_path}, use it.")
            score_func = 'unknown'
            label_type = 'unknown'
            for s in all_score_func:
                if s in custom_vc_path:
                    score_func = s
                    break
            for l in all_label_type:
                if l in custom_vc_path:
                    label_type = l
                    break
            for l in all_label_name:
                if l in custom_vc_path:
                    label_name = l
                    break
            all_score_func = [score_func]
            all_label_type = [label_type]
            all_label_name = [label_name]
            print(f"custom_vc_path: {custom_vc_path}, score_func: {score_func}, label_type: {label_type}, label_name: {label_name}")

        for label_name in all_label_name:
            for score_func in all_score_func:
                for label_type in all_label_type:
                    vc_path = custom_vc_path if custom_vc_path else get_vc_path(dst_name, dst_type, model_name, label_name, label_type, score_func)
                    if not os.path.exists(vc_path):
                        print(f"vc_path {vc_path} not exists, skip.")
                        continue
                    v_c.load_state_dict(torch.load(vc_path))
                    for v in v_c.values():
                        v.eval()
                        v.to(model.cfg.dtype).to(model.cfg.device)
                        for p in v.parameters():
                            p.requires_grad = False

                    print(f"Running get_uncertainty_score_ours_{score_func}_{label_type}_{label_name}")
                    ours_func = partial(get_uncertainty_score_ours_all, v_c=v_c, score_func=score_func, label_type=label_type, label_name=label_name, model=model)
                    test_dst = test_dst.map(ours_func, batched=True, batch_size=eval_batch_size, new_fingerprint=str(time()))

                    print(f"time_ours_{score_func}_{label_type}_{label_name}:{sum(test_dst[f'time_ours_{score_func}_{label_type}_{label_name}'])}")

    keys = (['options'] if test_dst[0].get('options') else []) + ['question', 'washed_answer', 'gt', 'num_answer_tokens'] + c_metrics + [k for k in test_dst[0].keys() if
                                                                                                                                         k.startswith("u_score") and not k.endswith("all")]
    for i in range(10):
        for k in keys:
            print(f"{k}:{test_dst[-i][k]}")
        print()

    # Save the result
    u_metrics = [k for k in test_dst[0].keys() if k.startswith("u_score") and not k.endswith("all")]
    for c_metric in c_metrics:
        fig = plot_th_curve(test_dst, u_metrics, c_metric)
        fig.write_html(f"{save_path}/{c_metric}_curve.html")
        fig.write_image(f"{save_path}/{c_metric}_curve.png")
    test_dst.save_to_disk(save_path)
    print(f"save all results to {save_path}")


if __name__ == "__main__":
    fire.Fire(evaluate)
