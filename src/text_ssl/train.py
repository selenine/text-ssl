import math
import os
import random
from dataclasses import asdict

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch import nn, optim
from torch.optim import swa_utils
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from text_ssl.configs import TrainConfig
from text_ssl.nn.model import Transformer


def train(
    model: Transformer,
    cfg: TrainConfig,
) -> None:
    accel = Accelerator(mixed_precision=cfg.mixed_precision, log_with="wandb")
    accel.init_trackers(
        project_name="text-ssl",
        config={"train": asdict(cfg), "model": asdict(model.cfg)},
        init_kwargs={"wandb": {"name": cfg.wandb_name}},
    )
    n_ctx = model.cfg.n_ctx

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

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

    model, dataloader, optimizer, scheduler = accel.prepare(
        model, dataloader, optimizer, scheduler
    )

    teacher = swa_utils.AveragedModel(
        accel.unwrap_model(model),
        multi_avg_fn=swa_utils.get_ema_multi_avg_fn(cfg.ema_wt),
        use_buffers=True,
    )
    teacher.requires_grad_(False)
    teacher.eval()

    criterion = nn.MSELoss()

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

        seq1, seq2 = get_seqs(batch)

        pred = model(seq1)
        with torch.no_grad(), accel.autocast():
            ref = teacher(seq2)

        loss = criterion(pred, ref)
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
                "train/lr": scheduler.get_last_lr()[0],
                "train/samples": (step + 1) * cfg.batch_size * accel.num_processes,
            }
            if grad_norm is not None:
                metrics["train/grad_norm"] = grad_norm
            accel.log(metrics, step=step)

        if cfg.save_every and (step + 1) % cfg.save_every == 0:
            save()

    save()

    accel.end_training()
