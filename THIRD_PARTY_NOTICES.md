# Third-party notices

Original code in this repository is (c) Jane Rose Ferdinant, licensed CC BY-NC-ND 4.0 (see `LICENSE`).
The components below are **not** redistributed here (data and weights are downloaded at run time) and keep
their own licenses.

## Dataset: PneumoniaMNIST (MedMNIST v2)

- Source: https://zenodo.org/records/10519652 (files `pneumoniamnist_64.npz`, `pneumoniamnist_128.npz`),
  project page https://medmnist.com/
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- Required attribution (please cite both):
  - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
    "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification."
    Scientific Data, 10, 41 (2023).
  - Daniel S. Kermany, Michael Goldbaum, Wenjia Cai, et al. "Identifying Medical Diagnoses and Treatable Diseases
    by Image-Based Deep Learning." Cell 172(5), 1122-1131 (2018). Source data: Mendeley Data, "Large Dataset of
    Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images", DOI 10.17632/rscbjbr9sj.3 (CC BY 4.0).
- Changes made by MedMNIST: images centre-cropped and resized. Changes made here: none to the files; the static
  explorer (`docs/data/`) contains 96 px thumbnails and Grad-CAM overlays of 120 test images derived from the
  128 px version. These derived images are shared under the dataset's CC BY 4.0 terms with the attribution above.
- File integrity: MD5 of `pneumoniamnist_128.npz` is `05b46931834c231683c68f40c47b2971`, checked at download.

## Pretrained model: MobileNetV3-Small ImageNet weights (torchvision)

- Weights file: `mobilenet_v3_small-047dcff4.pth` from https://download.pytorch.org/models/ , fetched at run time by
  torchvision into the local torch cache; never committed.
- Architecture and training recipe: Howard et al., "Searching for MobileNetV3", ICCV 2019.
- Weights are trained on ImageNet (https://www.image-net.org/); ImageNet is available for non-commercial research and
  educational use. This repository is non-commercial research and does not redistribute the weights.

## Software libraries (installed from PyPI, not vendored)

| Package | License | URL |
|---|---|---|
| PyTorch / torchvision | BSD-3-Clause | https://pytorch.org/ |
| NumPy | BSD-3-Clause | https://numpy.org/ |
| SciPy | BSD-3-Clause | https://scipy.org/ |
| scikit-learn | BSD-3-Clause | https://scikit-learn.org/ |
| Matplotlib | Matplotlib (PSF-based) license | https://matplotlib.org/ |
| Pillow | MIT-CMU (HPND) | https://python-pillow.org/ |
| pytest | MIT | https://pytest.org/ |
| Ruff | MIT | https://github.com/astral-sh/ruff |

## Method references (ideas, no code copied)

- Grad-CAM: Selvaraju et al., ICCV 2017. Temperature scaling: Guo et al., "On Calibration of Modern Neural Networks",
  ICML 2017. Focal loss: Lin et al., ICCV 2017.
