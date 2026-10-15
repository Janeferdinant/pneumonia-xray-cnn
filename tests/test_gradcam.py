import numpy as np
import torch

from pxray.gradcam import GradCAM, overlay_heatmap
from pxray.models import build_model, gradcam_layer
from pxray.shortcut import border_mass_fraction, keep_border_only, mask_border, ring_mask


def _model(name="small_cnn"):
    torch.manual_seed(0)
    return build_model(name, pretrained=False).eval()


def test_gradcam_output_shape_range_and_no_state_change():
    m = _model()
    x = torch.randn(4, 1, 64, 64)
    before = [p.clone() for p in m.parameters()]
    with GradCAM(m, gradcam_layer(m)) as cam:
        maps, logits = cam(x)
    assert maps.shape == (4, 64, 64)
    assert logits.shape == (4,)
    assert float(maps.min()) >= 0.0 and float(maps.max()) <= 1.0 + 1e-6
    for a, b in zip(before, m.parameters(), strict=True):
        assert torch.equal(a, b)
    assert all(p.grad is None for p in m.parameters())


def test_gradcam_is_deterministic():
    m = _model()
    x = torch.randn(2, 1, 64, 64)
    with GradCAM(m, gradcam_layer(m)) as cam:
        a, _ = cam(x)
        b, _ = cam(x)
    assert torch.equal(a, b)


def test_gradcam_mobilenet_shape():
    m = _model("mobilenet_v3_small")
    with GradCAM(m, gradcam_layer(m)) as cam:
        maps, _ = cam(torch.randn(2, 1, 64, 64))
    assert maps.shape == (2, 64, 64)


def test_gradcam_target_class_flips_evidence():
    m = _model()
    x = torch.randn(3, 1, 64, 64)
    with GradCAM(m, gradcam_layer(m)) as cam:
        pos, _ = cam(x, target="pneumonia")
        neg, _ = cam(x, target="normal")
    assert not torch.equal(pos, neg)


def test_hooks_removed_on_exit():
    m = _model()
    layer = gradcam_layer(m)
    n = len(layer._forward_hooks)
    with GradCAM(m, layer):
        assert len(layer._forward_hooks) == n + 1
    assert len(layer._forward_hooks) == n


def test_overlay_returns_uint8_rgb():
    img = (np.random.default_rng(0).random((32, 32)) * 255).astype(np.uint8)
    cam = np.random.default_rng(1).random((32, 32)).astype(np.float32)
    out = overlay_heatmap(img, cam, alpha=0.4)
    assert out.shape == (32, 32, 3) and out.dtype == np.uint8


def test_ring_mask_geometry_and_masking_variants():
    r = ring_mask(100, 100, 0.1)
    assert r.shape == (100, 100) and r.dtype == bool
    assert r[0, 50] and r[50, 0] and r[99, 50] and not r[50, 50]
    assert r.sum() == 100 * 100 - 80 * 80
    imgs = np.full((2, 100, 100), 100, np.uint8)
    imgs[:, :10, :] = 250
    centre = mask_border(imgs, 0.1, fill=7)
    assert (centre[:, r] == 7).all() and (centre[:, ~r] == 100).all()
    border = keep_border_only(imgs, 0.1, fill=7)
    assert (border[:, ~r] == 7).all() and (border[:, 0, 50] == 250).all()
    assert imgs[0, 0, 0] == 250  # inputs are not mutated


def test_border_mass_fraction():
    cam = np.zeros((1, 100, 100), np.float32)
    cam[0, :10, :] = 1.0
    f = border_mass_fraction(cam, 0.1)
    assert f.shape == (1,) and f[0] == 1.0
    uniform = np.ones((1, 100, 100), np.float32)
    assert abs(float(border_mass_fraction(uniform, 0.1)[0]) - 0.36) < 1e-6
