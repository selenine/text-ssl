import os
import random

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch import nn, optim
from torch.optim import swa_utils
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.text_ssl.nn.model import Transformer
from src.text_ssl.utils.configs import TrainConfig


def train(
    model: Transformer,
    cfg: TrainConfig,
) -> None:
    accel = Accelerator()
    n_ctx = model.cfg.n_ctx

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    dataset = (
        load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        .filter(lambda x: len(x["text"]) >= 2 * n_ctx)
        .map(lambda x: tokenizer(x["text"], truncation=True, max_length=2 * n_ctx))
        .with_format("torch")
    )

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    teacher = swa_utils.AveragedModel(
        model, swa_utils.get_ema_multi_avg_fn(cfg.ema_wt), use_buffers=True
    )

    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.wt_decay,
    )

    model, teacher, dataloader, optimizer = accel.prepare(
        model, teacher, dataloader, optimizer
    )

    criterion = nn.CrossEntropyLoss()

    def get_seqs(batch: torch.Tensor):
        a = random.randint(0, n_ctx)
        b = random.randint(0, n_ctx)

        return (batch[a : a + n_ctx], batch[b : b + n_ctx])

    batches = 0
    while True:
        batch = dataloader.next()

        seq1, seq2 = get_seqs(batch)

        pred = model(seq1)
        ref = teacher(seq2)

        loss = criterion(pred, ref)
        accel.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        batches += cfg.batch_size
        if batches > cfg.batch_size:
            break

    dir = os.path.dirname(cfg.save_path)
    if not os.path.exists(dir):
        os.mkdir(dir)

    torch.save(model, cfg.save_path)
