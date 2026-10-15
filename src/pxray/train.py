"""Seeded CPU training with best-validation-AUC checkpoint selection."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .augment import AugmentConfig, augment_batch
from .data import Split, Standardizer, pos_weight
from .losses import make_loss
from .metrics import roc_auc
from .models import build_model


@dataclass(frozen=True)
class TrainConfig:
    model: str = "small_cnn"
    pretrained: bool = False
    epochs: int = 30
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    loss: str = "bce"  # bce (class-weighted) | bce_unweighted | focal
    focal_gamma: float = 2.0
    augment: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    val_logits: np.ndarray
    test_logits: np.ndarray
    history: list[dict[str, float]]
    best_epoch: int
    seconds: float
    standardizer: Standardizer
    state_dict: dict = field(repr=False, default_factory=dict)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global seed on purpose, for any third-party use
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = [model(x[i : i + batch_size]) for i in range(0, len(x), batch_size)]
    return torch.cat(out).numpy().astype(np.float64)


def train_one(cfg: TrainConfig, splits: dict[str, Split], seed: int) -> RunResult:
    seed_everything(seed)
    t0 = time.time()
    st = Standardizer.fit(splits["train"].images)
    x_tr, x_va, x_te = (st.transform(splits[k].images) for k in ("train", "val", "test"))
    y_tr = torch.from_numpy(splits["train"].labels.astype(np.float32))
    y_va = splits["val"].labels

    model = build_model(cfg.model, pretrained=cfg.pretrained)
    loss_fn = make_loss(cfg.loss, pos_weight(splits["train"].labels), cfg.focal_gamma)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 0.5 * (1 + math.cos(math.pi * e / cfg.epochs)))
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    aug = AugmentConfig()

    best_auc, best_epoch, best_state = -1.0, 0, None
    history: list[dict[str, float]] = []
    n = len(x_tr)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = rng.permutation(n)
        total = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = torch.from_numpy(perm[i : i + cfg.batch_size])
            xb, yb = x_tr[idx], y_tr[idx]
            if len(idx) < 2:  # BatchNorm needs more than one sample
                continue
            if cfg.augment:
                xb = augment_batch(xb, gen, aug)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        val_auc = roc_auc(y_va, predict_logits(model, x_va))
        history.append({"epoch": epoch, "train_loss": total / n, "val_auc": val_auc})
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = copy.deepcopy(model.state_dict())

    assert best_state is not None
    model.load_state_dict(best_state)
    return RunResult(
        val_logits=predict_logits(model, x_va),
        test_logits=predict_logits(model, x_te),
        history=history,
        best_epoch=best_epoch,
        seconds=time.time() - t0,
        standardizer=st,
        state_dict=best_state,
    )
