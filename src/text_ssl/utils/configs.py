from dataclasses import dataclass


@dataclass
class TransformerConfig:
    n_ctx: int
    n_vocab: int
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_mlp: int


@dataclass
class TrainConfig:
    n_batches: int
    batch_size: int
    wandb_name: str
    lr: float
    wt_decay: float
    ema_wt: float
    save_path: str
