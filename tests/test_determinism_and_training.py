import numpy as np
import torch

from pxray.augment import AugmentConfig, augment_batch
from pxray.losses import focal_loss_with_logits, make_loss, weighted_bce
from pxray.metrics import roc_auc
from pxray.train import TrainConfig, seed_everything, train_one

from .conftest import make_toy_splits


def _cfg(**kw):
    base = {"model": "small_cnn", "pretrained": False, "epochs": 2, "batch_size": 16,
            "lr": 1e-3, "augment": True}
    base.update(kw)
    return TrainConfig(**base)


def test_same_seed_gives_identical_logits_and_history():
    s = make_toy_splits()
    a = train_one(_cfg(), s, seed=3)
    b = train_one(_cfg(), s, seed=3)
    assert np.array_equal(a.test_logits, b.test_logits)
    assert a.history == b.history


def test_different_seed_changes_result():
    s = make_toy_splits()
    a = train_one(_cfg(), s, seed=1)
    b = train_one(_cfg(), s, seed=2)
    assert not np.array_equal(a.test_logits, b.test_logits)


def test_training_learns_toy_task_better_than_chance():
    s = make_toy_splits(sizes=(200, 40, 80))
    r = train_one(_cfg(epochs=8), s, seed=0)
    assert roc_auc(s["test"].labels, r.test_logits) > 0.85


def test_seed_everything_reproduces_torch_and_numpy_draws():
    seed_everything(11)
    a = (torch.rand(3), np.random.rand(3))
    seed_everything(11)
    b = (torch.rand(3), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_augmentation_shape_seeded_and_never_flips():
    x = torch.zeros(8, 1, 32, 32)
    x[:, :, :8, :] = 1.0  # bright top band: a vertical flip would move it to the bottom
    cfg = AugmentConfig()
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    a, b = augment_batch(x, g1, cfg), augment_batch(x, g2, cfg)
    assert a.shape == x.shape and torch.equal(a, b)
    assert float(a[:, :, :16].mean()) > float(a[:, :, 16:].mean())
    assert not hasattr(cfg, "vertical_flip") and not hasattr(cfg, "horizontal_flip")


def test_augmentation_identity_when_ranges_zero():
    cfg = AugmentConfig(max_rotation_deg=0, scale=(1.0, 1.0), max_translate=0,
                        brightness=0, contrast=0)
    x = torch.randn(4, 1, 16, 16)
    out = augment_batch(x, torch.Generator().manual_seed(0), cfg)
    assert torch.allclose(out, x, atol=1e-5)


def test_losses():
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    plain = weighted_bce(logits, y, pos_weight=1.0)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    assert torch.isclose(plain, ref)
    assert weighted_bce(logits, y, pos_weight=3.0) > plain
    easy = focal_loss_with_logits(torch.tensor([8.0, -8.0]), torch.tensor([1.0, 0.0]), gamma=2.0)
    hard = focal_loss_with_logits(torch.tensor([-1.0, 1.0]), torch.tensor([1.0, 0.0]), gamma=2.0)
    assert easy < 1e-5 < hard
    assert torch.isclose(
        focal_loss_with_logits(logits, y, gamma=0.0, alpha=0.5), 0.5 * ref, atol=1e-6
    )
    assert callable(make_loss("bce", 2.0)) and callable(make_loss("focal", 2.0))
