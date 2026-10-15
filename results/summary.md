## Test-set performance (mean over seeds, bootstrap 95% CI over test images; seed SD in parentheses)

| Model | Seeds | ROC AUC | Sens. @ 90% spec | Sens. @ 95% spec | F1 @ 0.5 | Sensitivity @ 0.5 | Specificity @ 0.5 |
|---|---|---|---|---|---|---|---|
| Logistic regression (pixels) | 1 | 0.928 [0.903, 0.951] (0.000) | 0.808 [0.635, 0.898] (0.000) | 0.551 [0.282, 0.768] (0.000) | 0.907 [0.885, 0.927] (0.000) | 0.972 [0.955, 0.987] (0.000) | 0.714 [0.657, 0.774] (0.000) |
| Small CNN (from scratch) | 5 | 0.986 [0.978, 0.993] (0.003) | 0.985 [0.958, 0.996] (0.009) | 0.930 [0.836, 0.977] (0.021) | 0.964 [0.952, 0.975] (0.006) | 0.978 [0.966, 0.988] (0.008) | 0.914 [0.882, 0.944] (0.033) |
| MobileNetV3-Small (ImageNet, fine-tuned) | 5 | 0.990 [0.984, 0.994] (0.002) | 0.986 [0.957, 0.996] (0.004) | 0.942 [0.895, 0.976] (0.020) | 0.953 [0.939, 0.965] (0.005) | 0.993 [0.986, 0.999] (0.001) | 0.848 [0.807, 0.889] (0.018) |

## Other metrics at threshold 0.5 (mean over seeds)

| Model | Accuracy | Balanced acc. | Precision | Mean TN / FP / FN / TP |
|---|---|---|---|---|
| Logistic regression (pixels) | 0.875 | 0.843 | 0.850 | 167.0 / 67.0 / 11.0 / 379.0 |
| Small CNN (from scratch) | 0.954 | 0.946 | 0.950 | 213.8 / 20.2 / 8.6 / 381.4 |
| MobileNetV3-Small (ImageNet, fine-tuned) | 0.939 | 0.921 | 0.916 | 198.4 / 35.6 / 2.6 / 387.4 |

## Threshold chosen on validation (specificity >= 0.90 on validation), applied to test (mean over seeds)

| Model | Val-chosen threshold: test sensitivity | Test specificity | Validation AUC |
|---|---|---|---|
| Logistic regression (pixels) | 0.987 (0.000) | 0.611 (0.000) | 0.992 |
| Small CNN (from scratch) | 0.999 (0.001) | 0.662 (0.037) | 0.997 |
| MobileNetV3-Small (ImageNet, fine-tuned) | 1.000 (0.000) | 0.532 (0.022) | 0.999 |

## Calibration (test; temperature fitted on validation only)

| Model | T (mean, SD) | ECE raw [95% CI] | ECE after scaling [95% CI] | NLL raw -> scaled | Brier raw -> scaled |
|---|---|---|---|---|---|
| Logistic regression (pixels) | 0.82, 0.00 | 0.078 [0.061, 0.106] | 0.093 [0.074, 0.120] | 0.419 -> 0.467 | 0.100 -> 0.102 |
| Small CNN (from scratch) | 1.16, 0.28 | 0.026 [0.024, 0.042] | 0.027 [0.029, 0.043] | 0.144 -> 0.147 | 0.038 -> 0.038 |
| MobileNetV3-Small (ImageNet, fine-tuned) | 1.24, 0.16 | 0.043 [0.036, 0.062] | 0.039 [0.033, 0.058] | 0.222 -> 0.197 | 0.050 -> 0.049 |

## Paired differences (bootstrap 95% CI over test images)

| Comparison | ROC AUC | Sens. @ 90% spec | F1 @ 0.5 |
|---|---|---|---|
| MobileNetV3-Small (ImageNet, fine-tuned) minus Small CNN (from scratch) | 0.004 [-0.001, 0.009] | 0.001 [-0.017, 0.015] | -0.011 [-0.020, -0.001] |
| Small CNN (from scratch) minus Logistic regression (pixels) | 0.058 [0.038, 0.081] | 0.177 [0.089, 0.338] | 0.057 [0.039, 0.074] |
| MobileNetV3-Small (ImageNet, fine-tuned) minus Logistic regression (pixels) | 0.062 [0.041, 0.084] | 0.178 [0.090, 0.342] | 0.046 [0.031, 0.061] |

## Border-mask test (test AUC, mean over seeds, SD in parentheses)

Border ring width as a fraction of the image side. `centre` = ring masked (model sees the centre only); `border` = only the ring visible. Models were not retrained on masked inputs; logistic regression rows are separate classifiers trained on the masked pixels (data-level probe).

| Model | Original | centre 0.05 | centre 0.10 | centre 0.20 | border 0.05 | border 0.10 | border 0.20 |
|---|---|---|---|---|---|---|---|
| Logistic regression (pixels) | 0.928 (0.000) | 0.930 (0.000) | 0.929 (0.000) | 0.933 (0.000) | 0.800 (0.000) | 0.857 (0.000) | 0.908 (0.000) |
| Small CNN (from scratch) | 0.986 (0.003) | 0.978 (0.007) | 0.958 (0.012) | 0.856 (0.033) | 0.747 (0.071) | 0.871 (0.015) | 0.944 (0.014) |
| MobileNetV3-Small (ImageNet, fine-tuned) | 0.990 (0.002) | 0.982 (0.003) | 0.954 (0.014) | 0.735 (0.134) | 0.512 (0.112) | 0.611 (0.177) | 0.875 (0.052) |

## Ablations (seeds listed per block; paired bootstrap over test images)

Reference: Small CNN (from scratch); seeds [0, 1, 2]

| Variant | ROC AUC | Sensitivity @ 0.5 | Specificity @ 0.5 | F1 @ 0.5 | ECE raw |
|---|---|---|---|---|---|
| Small CNN (from scratch) (reference) | 0.987 [0.979, 0.993] | 0.979 [0.967, 0.990] | 0.913 [0.879, 0.945] | 0.964 [0.951, 0.975] | 0.027 [0.024, 0.045] |
| Small CNN, no class weights | 0.982 [0.973, 0.989] | 0.998 [0.995, 1.000] | 0.754 [0.703, 0.800] | 0.931 [0.914, 0.946] | 0.061 [0.049, 0.084] |
| Small CNN, focal loss | 0.984 [0.976, 0.991] | 0.979 [0.968, 0.990] | 0.893 [0.856, 0.926] | 0.959 [0.946, 0.971] | 0.088 [0.080, 0.104] |
| Small CNN, no augmentation | 0.976 [0.963, 0.987] | 0.996 [0.991, 0.999] | 0.792 [0.745, 0.840] | 0.939 [0.923, 0.954] | 0.052 [0.042, 0.074] |
| Small CNN, no class weights minus reference (difference) | -0.005 [-0.009, -0.003] | 0.020 [0.009, 0.031] | -0.160 [-0.195, -0.129] | -0.033 [-0.046, -0.022] | 0.034 [0.020, 0.048] |
| Small CNN, focal loss minus reference (difference) | -0.003 [-0.006, 0.001] | 0.001 [-0.005, 0.006] | -0.020 [-0.037, -0.006] | -0.005 [-0.011, -0.000] | 0.061 [0.038, 0.077] |
| Small CNN, no augmentation minus reference (difference) | -0.011 [-0.020, -0.004] | 0.017 [0.008, 0.027] | -0.121 [-0.156, -0.091] | -0.025 [-0.036, -0.014] | 0.024 [0.013, 0.039] |

Reference: MobileNetV3-Small (ImageNet, fine-tuned); seeds [0, 1, 2]

| Variant | ROC AUC | Sensitivity @ 0.5 | Specificity @ 0.5 | F1 @ 0.5 | ECE raw |
|---|---|---|---|---|---|
| MobileNetV3-Small (ImageNet, fine-tuned) (reference) | 0.990 [0.985, 0.995] | 0.993 [0.987, 0.998] | 0.840 [0.797, 0.880] | 0.951 [0.937, 0.964] | 0.046 [0.036, 0.066] |
| MobileNetV3-Small, random init | 0.970 [0.958, 0.981] | 0.993 [0.986, 0.998] | 0.544 [0.491, 0.599] | 0.876 [0.854, 0.898] | 0.106 [0.090, 0.134] |
| MobileNetV3-Small, random init minus reference (difference) | -0.020 [-0.029, -0.012] | 0.000 [-0.007, 0.007] | -0.296 [-0.343, -0.250] | -0.074 [-0.091, -0.060] | 0.060 [0.044, 0.078] |

