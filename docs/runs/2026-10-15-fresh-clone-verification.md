# Fresh-clone verification (2026-10-15)

Clone of `main` (commit 58b61c7) into an empty folder, new Python 3.11 venv, CPU PyTorch 2.14.1, Windows 11, 16 logical CPUs.

| README command | Result |
|---|---|
| `python -m pip install -r requirements.txt`, `python -m pip install -e . --no-deps` | installed |
| `ruff check . && ruff format --check .` | All checks passed, 37 files formatted |
| `pytest -q` (before download) | 42 passed, 1 skipped (real-data test) |
| `python -m pxray download` | file fetched, MD5 05b46931834c231683c68f40c47b2971 verified |
| `python -m pxray check-data` | sizes 4708 / 516 / 624, 8 validation duplicates removed, 0 cross-split duplicates |
| `python -m pxray train --seeds 0` (smoke: one seed instead of five) | small CNN seed 0: best epoch 12, val AUC 0.9967, 399 s; MobileNet seed 0: best epoch 8, val AUC 0.9989, 314 s; logistic regression done |
| `python -m pxray evaluate --n-boot 50` (smoke: 50 instead of 2000 resamples) | ran; tables and figures written |
| `python -m pxray explain`, `python -m pxray export-demo` | ran; explorer data 1.99 MB |
| `pytest -q` (after download) | 43 passed |

Not repeated in the check: the full five-seed training (about 1 h) and the three-seed ablations (about 1 h). Their logs are in `docs/runs/`.

Reproducibility: the seed-0 test logits retrained in the fresh clone and venv are bit-identical to the committed ones
for logistic regression, the small CNN and MobileNetV3-Small (maximum absolute difference 0.0).
