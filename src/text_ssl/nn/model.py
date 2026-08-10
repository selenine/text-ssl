import torch
from torch import nn

from src.text_ssl.nn.layers import MHSA, MLP
from src.text_ssl.utils.configs import TransformerConfig


class Transformer(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.n_vocab, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)

        self.register_buffer(
            "positions", torch.arange(cfg.n_ctx).unsqueeze(0), persistent=False
        )

        self.core = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.core.append(nn.LayerNorm(cfg.d_model))
            self.core.append(MHSA(cfg))
            self.core.append(nn.LayerNorm(cfg.d_model))
            self.core.append(MLP(cfg))

        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) + self.pos_embed(self.positions[:, : x.size(1)])
        x = self.core(x)

        return x
