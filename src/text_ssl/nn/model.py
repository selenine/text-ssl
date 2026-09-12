import torch
from torch import nn

from text_ssl.configs import TransformerConfig
from text_ssl.nn.layers import MHSA, MLP, DINOHead


class Transformer(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        super().__init__()

        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.n_vocab, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.dino_head = DINOHead(cfg)
        self.ibot_head = DINOHead(cfg)
        self.register_buffer(
            "positions", torch.arange(cfg.n_ctx).unsqueeze(0), persistent=False
        )

        self.core = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.core.append(nn.LayerNorm(cfg.d_model))
            self.core.append(MHSA(cfg))
            self.core.append(nn.LayerNorm(cfg.d_model))
            self.core.append(MLP(cfg))

    @torch.compile
    def forward(
        self, x: torch.Tensor, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.embedding(x) + self.pos_embed(self.positions[:, : x.size(1)])
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        for layer in self.core:
            x = layer(x)

        cls, toks = x[:, 0], x[:, 1:].flatten(0, 1).index_select(0, idx)

        return cls, self.dino_head(cls), self.ibot_head(toks)
