# Explorer data regeneration (2026-10-23)

The explorer files under `docs/data/` were never committed because the ignore rule `data/` also matched `docs/data/`.
The rule is now `/data/` (repository root only), and `git check-ignore -v` reports no ignored path under `docs/`.

Regeneration in a fresh clone and the pinned venv (PyTorch 2.14.1 CPU, 16 logical CPUs, Windows 11):

1. `python -m pxray download` (MD5 verified)
2. `python -m pxray --results artifacts/rerun/results --docs artifacts/rerun/docs train --run small_cnn mobilenet_v3_small --seeds 0`
   (separate output folder so the committed results and logs were not touched):
   small CNN seed 0: best epoch 12, val AUC 0.9967, 399 s; MobileNetV3-Small seed 0: best epoch 8, val AUC 0.9989, 306 s.
3. Check against the committed `results/predictions/{small_cnn,mobilenet_v3_small}_seed0.npz`: every array
   (validation logits, test logits and the six masked-input logit sets) is bit-identical for both models.
4. `python -m pxray export-demo`: `docs/data/explorer.json`, 120 thumbnails in `docs/data/img/` and 240 Grad-CAM overlays in
   `docs/data/cam/`, 361 files, 1.99 MB. The logits in the JSON match the committed test logits to the 4 decimals stored.

No result numbers changed. Thumbnails are derived from PneumoniaMNIST (CC BY 4.0, MedMNIST v2 / Kermany et al.); see THIRD_PARTY_NOTICES.md.
