import torch
from torch import nn
from torch.nn import functional as F

from text_ssl.configs import TransformerConfig


class MHSA(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
    ) -> None:
        super().__init__()

        self.cfg = cfg

        self.WQ = nn.Parameter(torch.zeros(cfg.d_model, cfg.n_heads * cfg.d_head))
        self.WK = nn.Parameter(torch.zeros(cfg.d_model, cfg.n_heads * cfg.d_head))
        self.WV = nn.Parameter(torch.zeros(cfg.d_model, cfg.n_heads * cfg.d_head))
        self.WO = nn.Parameter(torch.zeros(cfg.n_heads * cfg.d_head, cfg.d_model))

        for layer in [self.WQ, self.WK, self.WV, self.WO]:
            nn.init.kaiming_normal_(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (b, s, _), n, d = x.shape, self.cfg.n_heads, self.cfg.d_head
        q = (x @ self.WQ).view(b, s, n, d).transpose(1, 2)
        k = (x @ self.WK).view(b, s, n, d).transpose(1, 2)
        v = (x @ self.WV).view(b, s, n, d).transpose(1, 2)

        out = (
            F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, s, n * d)
            @ self.WO
        )

        return out + x


class MLP(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
    ) -> None:
        super().__init__()

        self.cfg = cfg

        self.Wup = nn.Parameter(torch.zeros(cfg.d_model, cfg.d_mlp))
        self.Wgate = nn.Parameter(torch.zeros(cfg.d_model, cfg.d_mlp))
        self.Wdown = nn.Parameter(torch.zeros(cfg.d_mlp, cfg.d_model))

        for layer in [self.Wup, self.Wgate, self.Wdown]:
            nn.init.kaiming_normal_(layer)

        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = x @ self.Wup, x @ self.Wgate
        out = self.act_fn(out * gate) @ self.Wdown

        return x + out
