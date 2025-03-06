from transformer_lens import HookedTransformer  # type: ignore

from trainer import Trainer
from utils import arg_parse_update_cfg, load_pile_lmsys_mixed_tokens


def get_model(name: str, device: str) -> HookedTransformer:
    return HookedTransformer.from_pretrained(name, device=device)


def get_default_cfg(base_model: HookedTransformer) -> dict:
    default_cfg = {
        "seed": 49,
        "batch_size": 4096,
        "buffer_mult": 128,
        "lr": 5e-5,
        "num_tokens": 400_000_000,
        "l1_coeff": 2,
        "beta1": 0.9,
        "beta2": 0.999,
        "d_in": base_model.cfg.d_model,
        "dict_size": 2**14,
        "seq_len": 1024,
        "enc_dtype": "fp32",
        "model_name": "gemma-2-2b",
        "site": "resid_pre",
        "device": "cuda:0",
        "model_batch_size": 4,
        "log_every": 100,
        "save_every": 30000,
        "dec_init_norm": 0.08,
        "hook_point": "blocks.14.hook_resid_pre",
        "wandb_project": "YOUR_WANDB_PROJECT",
        "wandb_entity": "YOUR_WANDB_ENTITY",
    }
    return arg_parse_update_cfg(default_cfg)


def get_default_trainer() -> Trainer:
    device = "cuda:0"
    base_model = get_model("gemma-2-2b", device)
    chat_model = get_model("gemma-2-2b-it", device)
    cfg = get_default_cfg(base_model)
    all_tokens = load_pile_lmsys_mixed_tokens()
    return Trainer(cfg, base_model, chat_model, all_tokens)
