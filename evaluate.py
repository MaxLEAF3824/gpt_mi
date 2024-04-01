import inspect
from utils import *

datasets.disable_caching()
torch.set_grad_enabled(False)

# All Config Variables
se_bert_name = "microsoft/deberta-large-mnli"
sentsim_bert_name = "all-MiniLM-L6-v2"
sar_bert_name = 'all-MiniLM-L6-v2'
all_score_func = ["mean", "last"]
all_label_type = ["soft"]
all_c_metric = ["rougel", "sentsim"]
all_u_metric = ["len", "pe", "sar", "ls", "se", "ours"]
eval_batch_size = 8
t_sar = 0.001


def evaluate(model_name,
             test_dst_path,
             c_metric="all",
             u_metric="all",
             rescale=False,
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

    # Load Test Dst    
    test_dst = Dataset.load_from_disk(test_dst_path)

    # test_dst = test_dst.select(range(120,150))

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
    if set(u_metrics) & {'ours', 'pe', 'sar'}:
        print("Running get_answer_prob")
        # get_answer_prob = partial(get_answer_prob, model=model)
        test_dst = test_dst.map(get_answer_prob, fn_kwargs=dict(model=model), batched=True, batch_size=eval_batch_size, new_fingerprint=str(time()))

    if set(u_metrics) & {'se', 'sar'}:
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
                ours_func = partial(get_uncertainty_score_ours_all, v_c=v_c, score_func=score_func, label_type=label_type, model=model)
                test_dst = test_dst.map(ours_func, batched=True, batch_size=eval_batch_size, new_fingerprint=str(time()))
                print(f"time_ours_{score_func}_{label_type}:{sum(test_dst[f'time_ours_{score_func}_{label_type}'])}")

    if rescale:
        for k in test_dst.column_names:
            if k.startswith("u_score") and not k.endswith("all"):
                mean = np.mean(test_dst[k])
                std = np.std(test_dst[k])
                test_dst = test_dst.map(partial(rescale_uscore, u_metric=k, mean=mean, std=std), new_fingerprint=str(time()))

    keys = (['options'] if test_dst[0].get('options') else []) + ['question', 'washed_answer', 'gt', 'num_answer_tokens'] + c_metrics + [k for k in test_dst[0].keys() if
                                                                                                                                         k.startswith("u_score") and not k.endswith("all")]
    for i in range(10):
        for k in keys:
            print(f"{k}:{test_dst[-i][k]}")
        print()

    # Save the result
    base_dir = "/mnt/petrelfs/guoyiqiu/coding/trainable_uncertainty/eval_results"
    save_base_name = f"{base_dir}/{model_name}/{dst_name}_{dst_type}"
    os.makedirs(save_base_name, exist_ok=True)

    u_metrics = [k for k in test_dst[0].keys() if k.startswith("u_score") and not k.endswith("all")]

    for c_metric in c_metrics:
        fig = plot_th_curve(test_dst, u_metrics, c_metric)
        fig.write_html(f"{save_base_name}/{c_metric}_curve.html")
        fig.write_image(f"{save_base_name}/{c_metric}_curve.png")

    test_dst.save_to_disk(save_base_name)
    print(f"save all results to {save_base_name}")


if __name__ == "__main__":
    fire.Fire(evaluate)
