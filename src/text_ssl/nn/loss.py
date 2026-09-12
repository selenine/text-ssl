import torch
from accelerate.utils import reduce
from torch import nn
from torch.nn import functional as F

type ModelOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class DINOLoss(nn.Module):
    def __init__(
        self,
        student_temp: float,
        n_sk_iters: int = 3,
    ) -> None:
        super().__init__()

        self.student_temp = student_temp
        self.n_sk_iters = n_sk_iters

    @torch.no_grad()
    def sinkhorn_knopp(self, logits: torch.Tensor, temp: float) -> torch.Tensor:
        Q = torch.exp(logits.float().flatten(0, -2) / temp).T
        K = Q.size(0)
        B = reduce(torch.tensor(float(Q.size(1)), device=Q.device), "sum")

        Q /= reduce(Q.sum(), "sum")
        for _ in range(self.n_sk_iters):
            Q /= reduce(Q.sum(dim=1, keepdim=True), "sum")
            Q /= K

            Q /= Q.sum(dim=0, keepdim=True)
            Q /= B

        Q *= B

        return Q.T

    def cross_entropy(self, t_probs: torch.Tensor, s_logits: torch.Tensor) -> torch.Tensor:
        s_logprobs = F.log_softmax(s_logits.float().flatten(0, -2) / self.student_temp, dim=-1)

        return -(t_probs * s_logprobs).sum(dim=-1)

    def koleo(self, feats: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        feats = F.normalize(feats.float(), dim=-1, eps=eps)

        with torch.no_grad():
            dots = feats @ feats.T
            dots.fill_diagonal_(-1)
            nn_idx = dots.argmax(dim=1)

        dists = F.pairwise_distance(feats, feats[nn_idx], eps=eps)

        return -torch.log(dists + eps).mean()

    def forward(
        self,
        student: ModelOutput,
        teacher: ModelOutput,
        teacher_temp: float,
        weights: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        s_feats, s_cls, s_tok = student
        _, t_cls, t_tok = teacher

        s_cls1, s_cls2 = s_cls.chunk(2)
        t_cls1, t_cls2 = self.sinkhorn_knopp(t_cls, teacher_temp).chunk(2)
        dino = (
            self.cross_entropy(t_cls2, s_cls1).mean()
            + self.cross_entropy(t_cls1, s_cls2).mean()
        ) / 2

        t_tok = self.sinkhorn_knopp(t_tok, teacher_temp)
        ibot = (self.cross_entropy(t_tok, s_tok) * weights).sum() / s_cls.size(0)

        koleo = sum(self.koleo(f) for f in s_feats.chunk(2)) / 2

        return {"dino": dino, "ibot": ibot, "koleo": koleo}
