import math
import os
import random
from dataclasses import asdict

import torch
from accelerate import Accelerator
from accelerate.utils import AORecipeKwargs
from datasets import load_dataset
from torch import optim
from torch.optim import swa_utils
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from text_ssl.configs import TrainConfig
from text_ssl.nn.loss import DINOLoss
from text_ssl.nn.model import Transformer


def get_masks(
    n_seqs: int, n_ctx: int, cfg: TrainConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_masked_seqs = int(n_seqs * cfg.mask_prob)
    probs = torch.linspace(cfg.mask_ratio_min, cfg.mask_ratio_max, n_masked_seqs + 1)
    counts = [int(n_ctx * p) for p in probs[1:]] + [0] * (n_seqs - n_masked_seqs)
    random.shuffle(counts)

    ranks = torch.rand(n_seqs, n_ctx).argsort(dim=1).argsort(dim=1)
    mask = ranks < torch.tensor(counts)[:, None]

    idx = mask.flatten().nonzero().flatten()
    weights = (1 / mask.sum(dim=-1).clamp(min=1))[:, None].expand_as(mask)[mask]

    return mask, idx, weights


def train(
    model: Transformer,
    cfg: TrainConfig,
) -> None:
    kwargs_handlers = []
    if cfg.mixed_precision == "fp8":
        kwargs_handlers.append(
            AORecipeKwargs(module_filter_func=lambda _, fqn: fqn.startswith("core."))
        )

    accel = Accelerator(
        mixed_precision=cfg.mixed_precision,
        log_with="wandb",
        kwargs_handlers=kwargs_handlers,
    )
    accel.init_trackers(
        project_name="text-ssl",
        config={"train": asdict(cfg), "model": asdict(model.cfg)},
        init_kwargs={"wandb": {"name": cfg.wandb_name}},
    )
    n_ctx = model.cfg.n_ctx

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    mask_token_id = tokenizer.mask_token_id

    dataset = (
        load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        .map(
            lambda x: {
                "input_ids": tokenizer(
                    x["text"], truncation=True, max_length=2 * n_ctx
                )["input_ids"]
            },
            remove_columns=["text"],
        )
        .filter(lambda x: len(x["input_ids"]) == 2 * n_ctx)
        .shuffle(seed=0, buffer_size=10_000)
        .with_format("torch")
    )

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size)

    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.wt_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < cfg.n_warmup:
            return (step + 1) / max(cfg.n_warmup, 1)

        progress = (step - cfg.n_warmup) / max(cfg.n_batches - cfg.n_warmup, 1)

        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    criterion = DINOLoss(cfg.student_temp)

    def teacher_temp(step: int) -> float:
        if step >= cfg.n_teacher_warmup:
            return cfg.teacher_temp

        progress = step / max(cfg.n_teacher_warmup, 1)

        return cfg.teacher_temp_warmup + progress * (cfg.teacher_temp - cfg.teacher_temp_warmup)

    def teacher_momentum(step: int) -> float:
        progress = min(step / max(cfg.n_batches, 1), 1.0)
        start, end = cfg.teacher_momentum, cfg.teacher_momentum_final

        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))

    model, dataloader, optimizer, scheduler = accel.prepare(
        model, dataloader, optimizer, scheduler
    )

    def ema_avg(
        t_params: list[torch.Tensor], s_params: list[torch.Tensor], n_averaged: torch.Tensor
    ) -> None:
        torch._foreach_lerp_(t_params, s_params, 1.0 - teacher_momentum(n_averaged.item()))

    teacher = swa_utils.AveragedModel(accel.unwrap_model(model), multi_avg_fn=ema_avg)
    teacher.requires_grad_(False)
    teacher.eval()

    def get_batches():
        while True:
            yield from dataloader

    def get_seqs(batch: dict[str, torch.Tensor]):
        ids = batch["input_ids"]
        a = random.randint(0, n_ctx)
        b = random.randint(0, n_ctx)

        return (ids[:, a : a + n_ctx], ids[:, b : b + n_ctx])

    def save() -> None:
        accel.wait_for_everyone()
        if not accel.is_main_process:
            return

        dir = os.path.dirname(cfg.save_path)
        if dir:
            os.makedirs(dir, exist_ok=True)

        accel.save(accel.unwrap_model(model).state_dict(), cfg.save_path)

    batches = get_batches()
    for step in range(cfg.n_batches):
        batch = next(batches)

        seqs = torch.cat(get_seqs(batch))

        mask, idx, weights = (
            t.to(accel.device) for t in get_masks(seqs.size(0), n_ctx, cfg)
        )
        masked = seqs.masked_fill(mask, mask_token_id)

        student_out = model(masked, idx)
        with torch.no_grad(), accel.autocast():
            teacher_out = teacher(seqs, idx)

        losses = criterion(student_out, teacher_out, teacher_temp(step), weights)
        loss = (
            losses["dino"]
            + cfg.mask_wt * losses["ibot"]
            + cfg.koleo_wt * losses["koleo"]
        )
        accel.backward(loss)

        grad_norm = None
        if cfg.grad_norm and accel.sync_gradients:
            grad_norm = accel.clip_grad_norm_(model.parameters(), cfg.grad_norm).item()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        teacher.update_parameters(accel.unwrap_model(model))

        if step % cfg.log_every == 0:
            metrics = {
                "train/loss": loss.item(),
                **{f"train/{k}_loss": v.item() for k, v in losses.items()},
                "train/lr": scheduler.get_last_lr()[0],
                "train/teacher_temp": teacher_temp(step),
                "train/teacher_momentum": teacher_momentum(step),
                "train/samples": (step + 1) * cfg.batch_size * accel.num_processes,
            }
            if grad_norm is not None:
                metrics["train/grad_norm"] = grad_norm
            accel.log(metrics, step=step)

        if cfg.save_every and (step + 1) % cfg.save_every == 0:
            save()

    save()

    accel.end_training()
