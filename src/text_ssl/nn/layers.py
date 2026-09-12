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

        self.WQ = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.WK = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.WV = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.WO = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

        for layer in [self.WQ, self.WK, self.WV, self.WO]:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (b, s, _), n, d = x.shape, self.cfg.n_heads, self.cfg.d_head
        q = self.WQ(x).view(b, s, n, d).transpose(1, 2)
        k = self.WK(x).view(b, s, n, d).transpose(1, 2)
        v = self.WV(x).view(b, s, n, d).transpose(1, 2)

        out = self.WO(
            F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, s, n * d)
        )

        return out + x


class MLP(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
    ) -> None:
        super().__init__()

        self.cfg = cfg

        self.Wup = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.Wgate = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.Wdown = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)

        for layer in [self.Wup, self.Wgate, self.Wdown]:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = self.Wup(x), self.Wgate(x)
        out = self.Wdown(F.silu(out * gate))

        return x + out


class DINOHead(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_proj),
            nn.GELU(),
            nn.Linear(cfg.d_proj, cfg.d_proj),
            nn.GELU(),
            nn.Linear(cfg.d_proj, cfg.d_bottleneck),
        )
        self.prototypes = nn.Linear(cfg.d_bottleneck, cfg.d_reps, bias=False)

        for layer in [*self.mlp, self.prototypes]:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prototypes(F.normalize(self.mlp(x), dim=-1))
