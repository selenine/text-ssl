import torch
from torch import nn
from torch.nn import functional as F

from src.text_ssl.utils.configs import TransformerConfig


class MHSA(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
    ) -> None:
        self.cfg = cfg

        self.WQ = nn.Parameter(torch.zeros(cfg.d_model, cfg.n_heads * cfg.d_head))
        self.WK = nn.Parameter(torch.zeros(cfg.d_model, cfg.n_heads * cfg.d_head))
        self.WV = nn.Parameter(torch.zeros(cfg.d_model, cfg.n_heads * cfg.d_head))
        self.WO = nn.Parameter(torch.zeros(cfg.n_heads * cfg.d_head, cfg.d_model))

        for layer in [self.WQ, self.WK, self.WV, self.WO]:
            nn.init.kaiming_normal_(layer)

        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (b, s, d), n = x.shape, self.cfg.n_heads
        q = (x @ self.WQ).view(b, s, n, d)
        k = (x @ self.WK).view(b, s, n, d)
        v = (x @ self.WV).view(b, s, n, d)

        pre = []
        for i in range(n):
            pre.append(F.softmax(q[:, :, i] @ k[:, :, i].T) @ v[:, :, i])

        pre = torch.tensor(pre).view(b, s, n * d)
        out = pre @ self.WO

        return out + x


class MLP(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
    ) -> None:
        self.cfg = cfg

        self.Wup = nn.Parameter(torch.zeros(cfg.d_model, cfg.d_mlp))
        self.Wgate = nn.Parameter(torch.zeros(cfg.d_model, cfg.d_mlp))
        self.Wdown = nn.Parameter(torch.zeros(cfg.d_mlp, cfg.d_model))

        for layer in [self.Wup, self.Wgate, self.Wdown]:
            nn.init.kaiming_normal_(layer)

        self.act_fn = nn.SiLU()

        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = x @ self.Wup, x @ self.Wgate
        out = self.act_fn(out * gate) @ self.Wdown

        return x + out
