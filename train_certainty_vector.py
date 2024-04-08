import wandb
import inspect
from utils import *

datasets.disable_caching()
torch.set_grad_enabled(False)
print_sys_info()

nli_pipe_name = "microsoft/deberta-large-mnli"
sentsim_bert_name = "all-MiniLM-L6-v2"


def train_certainty_vector(
        model_name,
        train_dst_path: str,
        val_dst_path: str,
        c_metric: str,
        c_th: float,
        score_func: str,
        lr: float,
        epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        max_train_data_size: int,
        max_val_data_size: int,
        label_type: str,
        layers: Union[int, str],
        act_name: str,
        
):
    args, _, _, arg_values = inspect.getargvalues(inspect.currentframe())
    for arg in args:
        print(f"{arg} = {arg_values[arg]}")
    
    all_c_metric = ["rougel", "sentsim", "include"]
    if c_metric not in all_c_metric:
        raise ValueError(f"c_metric {c_metric} not supported")
    
    train_dst_name = train_dst_path.split("/")[-1].split("_")[0]
    train_dst_type = "short" if "short" in train_dst_path else "long"

    val_dst_name = val_dst_path.split("/")[-1].split("_")[0]
    val_dst_type = "short" if "short" in val_dst_path else "long"

    print(f"train_dst_name = {train_dst_name}")
    print(f"val_dst_type = {val_dst_type}")

    # Model Config
    hooked_transformer_name = get_hooked_transformer_name(model_name)
    hf_model_path = os.path.join(os.environ["my_models_dir"], model_name)

    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    with LoadWoInit():
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)

    model = HookedTransformer.from_pretrained_no_processing(hooked_transformer_name, dtype='bfloat16', hf_model=hf_model, tokenizer=hf_tokenizer, default_padding_side='left')

    # Data Config
    train_dst = Dataset.load_from_disk(train_dst_path)
    val_dst = Dataset.load_from_disk(val_dst_path)

    if max_train_data_size < len(train_dst):
        train_dst = train_dst.select(list(range(max_train_data_size)))
    if max_val_data_size < len(val_dst):
        val_dst = val_dst.select(list(range(max_val_data_size)))

    train_data_size = len(train_dst)
    val_data_size = len(val_dst)

    print("Running wash_answer")
    train_dst = train_dst.map(partial(wash_answer, tokenizer=hf_tokenizer), new_fingerprint=str(time()))
    val_dst = val_dst.map(partial(wash_answer, tokenizer=hf_tokenizer), new_fingerprint=str(time()))

    if c_metric == 'rougel':
        print("Running get_rougel")
        train_dst = train_dst.map(get_rougel, new_fingerprint=str(time()))
        val_dst = val_dst.map(get_rougel, new_fingerprint=str(time()))
    elif c_metric == 'sentsim':
        st_model = SentenceTransformer(sentsim_bert_name)
        train_dst = train_dst.map(partial(get_sentsim, st_model=st_model), batched=True, batch_size=2, new_fingerprint=str(time()))
        val_dst = val_dst.map(partial(get_sentsim, st_model=st_model), batched=True, batch_size=2, new_fingerprint=str(time()))
    elif c_metric == 'include':
        print("Running get_include")
        train_dst = train_dst.map(get_include, new_fingerprint=str(time()))
        val_dst = val_dst.map(get_include, new_fingerprint=str(time()))
    else:
        raise ValueError(f"metric {c_metric} not supported")

    keys = (['options'] if val_dst[0].get('options') else []) + ['question', 'washed_answer', 'gt', c_metric]
    for i in range(10):
        for k in keys:
            print(f"{k}:{val_dst[i][k]}")
        print()

    torch.set_grad_enabled(True)
    model.requires_grad_(False)

    if layers == "all":
        layers = list(range(0, model.cfg.n_layers))
    elif isinstance(layers, int):
        layers = [layers]
    else:
        raise ValueError(f"layers {layers} not supported")

    if act_name not in ['resid_post', 'hook_attn_out', 'hook_mlp_out']:
        raise ValueError(f"act_name {act_name} not supported")
    full_act_names = [utils.get_act_name(act_name, l) for l in sorted(layers)]

    # module config
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
    v_c.to(model.cfg.dtype).to(model.cfg.device)

    print("Running get_num_tokens")
    train_dst = train_dst.map(get_num_tokens, new_fingerprint=str(time()), batched=True, batch_size=batch_size)
    val_dst = val_dst.map(get_num_tokens, new_fingerprint=str(time()), batched=True, batch_size=batch_size)

    # setup optimizer
    optimizer = torch.optim.Adam(v_c.parameters(), lr=lr)

    # setup progress bar and wandb
    bar = tqdm(total=(math.ceil(len(train_dst) / batch_size) + math.ceil(len(val_dst) / batch_size)) * epochs, unit='step')
    wandb.init(project='uncertainty', name=f"{train_dst_name}_{c_metric} ", tags=[score_func, model_name, label_type])
    wandb.log(arg_values)
    best_auroc = 0

    def loss_func(batch_scores, batch_labels):
        if label_type == 'hard':
            batch_labels = [1 if l > c_th else 0 for l in batch_labels]
        elif label_type == 'soft':
            pass
        else:
            raise ValueError(f"label_type {label_type} not supported")
        batch_labels = torch.tensor(batch_labels, dtype=batch_scores.dtype).to(batch_scores.device)
        loss = F.mse_loss(batch_scores, batch_labels)
        if loss.isnan():
            print(f"batch_scores:{batch_scores}")
        return loss

    def eval_func(scores, labels):
        discrete_labels = [1 if l > c_th else 0 for l in labels]
        return roc_auc_score(discrete_labels, scores)

    save_dir = f"models/{model_name}/{c_metric}"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        epoch_log = {}
        for phase in ['train', 'val']:
            epoch_loss = []
            epoch_scores = []

            if phase == 'train':
                random.seed(42 + epoch)
                dst = train_dst.shuffle(seed=42 + epoch)
                v_c.train()
            else:
                dst = val_dst
                v_c.eval()

            for i in range(0, len(dst), batch_size):
                batch = dst[i:i + batch_size]

                if phase == 'train':
                    batch_scores = ours_forward_func(batch, v_c, score_func, model)
                else:
                    with torch.no_grad():
                        batch_scores = ours_forward_func(batch, v_c, score_func, model)

                loss = loss_func(batch_scores, batch[c_metric])

                if phase == 'train' and i % gradient_accumulation_steps == 0:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    wandb.log({f'train_step_loss': loss.item()})
                
                epoch_loss.append(loss.item())
                epoch_scores.extend(batch_scores.tolist())

                bar.update(1)
                bar.set_description(f'{phase} loss:{loss.item():.4f}', refresh=True)

            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            epoch_auroc = eval_func(epoch_scores, dst[c_metric])
            epoch_log.update({f'{phase}_loss': epoch_loss, f'{phase}_auroc': epoch_auroc})

        wandb.log(epoch_log)
        print(f"epoch {epoch} log:{epoch_log}")

        if epoch_log['val_auroc'] > best_auroc:
            best_auroc = epoch_log['val_auroc']
            save_name = f"v_c_{train_dst_name}_{train_dst_type}_{score_func}_{label_type}_best.pth"
            torch.save(v_c.state_dict(), f"{save_dir}/{save_name}")
            print(f"new best auroc:{best_auroc}")
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(train_certainty_vector)
